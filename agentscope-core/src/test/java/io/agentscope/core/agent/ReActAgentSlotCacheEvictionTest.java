/*
 * Copyright 2024-2026 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.agentscope.core.agent;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.agentscope.core.ReActAgent;
import io.agentscope.core.message.ContentBlock;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.MsgRole;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.model.ChatResponse;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.ToolSchema;
import io.agentscope.core.state.InMemoryAgentStateStore;
import java.lang.reflect.Field;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Flux;

/**
 * Locks the observable contract of the bounded state/permission-engine slot caches on {@link
 * ReActAgent}: a 1,000-slot bound, least-recently-used eviction ordering, and paired removal of a
 * slot's {@code AgentState} and {@code PermissionEngine} together.
 *
 * <p>Note: {@code GracefulShutdownMiddleware} refreshes the agent's no-arg default-session slot
 * (via {@code checkAndClearShutdownInterrupted}) on every {@code call()}, which permanently keeps
 * one bookkeeping entry warm in {@code stateCache} (but never in {@code permissionEngineCache},
 * since that accessor never touches the permission cache). Tests that drive the cache via {@code
 * call()} account for that extra, always-fresh slot instead of asserting exact slot counts.
 */
@DisplayName("ReActAgent bounded slot cache eviction")
class ReActAgentSlotCacheEvictionTest {

    private static final int MAX_CACHED_SLOTS = 1000;

    private static final class NoopModel extends ChatModelBase {
        @Override
        public String getModelName() {
            return "noop";
        }

        @Override
        protected Flux<ChatResponse> doStream(
                List<Msg> messages, List<ToolSchema> tools, GenerateOptions options) {
            return Flux.just(
                    ChatResponse.builder()
                            .content(List.<ContentBlock>of(TextBlock.builder().text("ok").build()))
                            .build());
        }
    }

    private ReActAgent agent(InMemoryAgentStateStore store) {
        return ReActAgent.builder()
                .name("asst")
                .sysPrompt("hi")
                .model(new NoopModel())
                .stateStore(store)
                .build();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> cacheField(ReActAgent agent, String fieldName)
            throws Exception {
        Field f = ReActAgent.class.getDeclaredField(fieldName);
        f.setAccessible(true);
        return (Map<String, Object>) f.get(agent);
    }

    private static void callSlot(ReActAgent agent, String sessionId) {
        RuntimeContext ctx = RuntimeContext.builder().userId("u").sessionId(sessionId).build();
        agent.call(List.of(userMsg("hello")), ctx).block(Duration.ofSeconds(10));
    }

    private static Msg userMsg(String text) {
        return Msg.builder()
                .name("user")
                .role(MsgRole.USER)
                .content(TextBlock.builder().text(text).build())
                .build();
    }

    @Test
    @DisplayName("state cache never grows past the 1,000-slot bound")
    void stateCacheStaysBoundedAt1000Slots() throws Exception {
        InMemoryAgentStateStore store = new InMemoryAgentStateStore();
        ReActAgent agent = agent(store);

        int totalSlots = MAX_CACHED_SLOTS + 50;
        for (int i = 0; i < totalSlots; i++) {
            agent.getAgentState("u", "sess-" + i);
        }

        Map<String, Object> stateCache = cacheField(agent, "stateCache");
        assertTrue(
                stateCache.size() <= MAX_CACHED_SLOTS,
                "state cache must never exceed the configured bound; was " + stateCache.size());
    }

    @Test
    @DisplayName("a re-accessed slot is protected from eviction (LRU ordering)")
    void reaccessedSlotIsProtectedFromEviction() throws Exception {
        InMemoryAgentStateStore store = new InMemoryAgentStateStore();
        ReActAgent agent = agent(store);

        // Fill the cache to the bound via distinct calls, which populate both the state and
        // permission-engine caches together for each activated slot.
        for (int i = 0; i < MAX_CACHED_SLOTS; i++) {
            callSlot(agent, "s-" + i);
        }

        Map<String, Object> stateCache = cacheField(agent, "stateCache");
        Map<String, Object> permissionEngineCache = cacheField(agent, "permissionEngineCache");
        assertTrue(stateCache.size() <= MAX_CACHED_SLOTS);
        assertTrue(permissionEngineCache.size() <= MAX_CACHED_SLOTS);
        assertFalse(stateCache.containsKey("u/s-0"), "the oldest slot should already be evicted");
        assertTrue(stateCache.containsKey("u/s-1"), "sanity: slot 1 should still be cached");
        assertTrue(
                permissionEngineCache.containsKey("u/s-1"),
                "sanity: slot 1's permission engine should still be cached");

        // Refresh slot "s-1" so it becomes the most-recently-used entry.
        callSlot(agent, "s-1");

        // Push the cache one entry past the bound with a brand-new slot; the least-recently-used
        // surviving entry ("s-2", never touched again) must be evicted, not the refreshed "s-1".
        callSlot(agent, "s-overflow");

        assertTrue(stateCache.size() <= MAX_CACHED_SLOTS);
        assertTrue(permissionEngineCache.size() <= MAX_CACHED_SLOTS);
        assertTrue(
                stateCache.containsKey("u/s-1"), "recently re-accessed slot must survive eviction");
        assertTrue(
                permissionEngineCache.containsKey("u/s-1"),
                "recently re-accessed slot's permission engine must survive eviction");
        assertFalse(
                stateCache.containsKey("u/s-2"), "the least-recently-used slot must be evicted");
        assertFalse(
                permissionEngineCache.containsKey("u/s-2"),
                "eviction must remove the paired permission engine as well");
        assertTrue(
                stateCache.containsKey("u/s-overflow"), "the newly activated slot must be cached");
        assertTrue(
                permissionEngineCache.containsKey("u/s-overflow"),
                "the newly activated slot's permission engine must be cached");
    }

    @Test
    @DisplayName("eviction always removes the state and permission-engine entries together")
    void evictionRemovesStateAndPermissionEnginePaired() throws Exception {
        InMemoryAgentStateStore store = new InMemoryAgentStateStore();
        ReActAgent agent = agent(store);

        int totalSlots = MAX_CACHED_SLOTS + 25;
        for (int i = 0; i < totalSlots; i++) {
            callSlot(agent, "p-" + i);
        }

        Map<String, Object> stateCache = cacheField(agent, "stateCache");
        Map<String, Object> permissionEngineCache = cacheField(agent, "permissionEngineCache");
        assertTrue(stateCache.size() <= MAX_CACHED_SLOTS);
        assertTrue(permissionEngineCache.size() <= MAX_CACHED_SLOTS);

        // A permission engine must never outlive the state entry it was paired with: whatever
        // remains cached for permissions must also still be cached for state.
        assertTrue(
                stateCache.keySet().containsAll(permissionEngineCache.keySet()),
                "every cached permission engine must have a paired cached state entry");

        // The oldest slot must be evicted from both caches; the newest must remain in both.
        assertFalse(stateCache.containsKey("u/p-0"), "oldest slot should be evicted from state");
        assertFalse(
                permissionEngineCache.containsKey("u/p-0"),
                "oldest slot should be evicted from permission cache");
        String newest = "u/p-" + (totalSlots - 1);
        assertTrue(stateCache.containsKey(newest), "newest slot should remain cached in state");
        assertTrue(
                permissionEngineCache.containsKey(newest),
                "newest slot should remain cached in permission engine cache");
    }

    @Test
    @DisplayName("concurrent activation across many sessions stays within bound without corruption")
    void concurrentActivationStaysBoundedAndPaired() throws Exception {
        InMemoryAgentStateStore store = new InMemoryAgentStateStore();
        ReActAgent agent = agent(store);

        int sessions = MAX_CACHED_SLOTS + 200;
        ExecutorService pool = Executors.newFixedThreadPool(16);
        AtomicInteger failures = new AtomicInteger();
        try {
            List<Future<?>> futures =
                    IntStream.range(0, sessions)
                            .mapToObj(
                                    i ->
                                            pool.submit(
                                                    () -> {
                                                        try {
                                                            callSlot(agent, "c-" + i);
                                                        } catch (Exception e) {
                                                            failures.incrementAndGet();
                                                        }
                                                    }))
                            .collect(Collectors.toList());
            for (Future<?> f : futures) {
                f.get();
            }
        } finally {
            pool.shutdown();
        }

        assertEquals(0, failures.get(), "no concurrent activation should throw");

        Map<String, Object> stateCache = cacheField(agent, "stateCache");
        Map<String, Object> permissionEngineCache = cacheField(agent, "permissionEngineCache");
        assertTrue(
                stateCache.size() <= MAX_CACHED_SLOTS,
                "state cache must stay within bound under concurrency; was " + stateCache.size());
        assertTrue(
                permissionEngineCache.size() <= MAX_CACHED_SLOTS,
                "permission-engine cache must stay within bound under concurrency; was "
                        + permissionEngineCache.size());
        assertTrue(
                stateCache.keySet().containsAll(permissionEngineCache.keySet()),
                "every cached permission engine must have a paired cached state entry, even after"
                        + " concurrent eviction");
    }

    @Test
    @DisplayName("getAgentState without a configured store still enforces the bound")
    void getAgentStateWithoutStoreEnforcesBound() throws Exception {
        ReActAgent agent =
                ReActAgent.builder().name("asst").sysPrompt("hi").model(new NoopModel()).build();

        int totalSlots = MAX_CACHED_SLOTS + 10;
        for (int i = 0; i < totalSlots; i++) {
            agent.getAgentState("u", "nostore-" + i);
        }

        Map<String, Object> stateCache = cacheField(agent, "stateCache");
        assertTrue(
                stateCache.size() <= MAX_CACHED_SLOTS,
                "state cache must stay bounded even without a configured store; was "
                        + stateCache.size());
        assertFalse(
                stateCache.containsKey("u/nostore-0"), "oldest untouched slot should be evicted");
        assertTrue(
                stateCache.containsKey("u/nostore-" + (totalSlots - 1)),
                "most recently created slot must remain cached");
    }
}
