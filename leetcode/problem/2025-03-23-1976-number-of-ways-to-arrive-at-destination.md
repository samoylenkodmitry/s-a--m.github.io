---
layout: leetcode-entry
title: "1976. Number of Ways to Arrive at Destination"
permalink: "/leetcode/problem/2025-03-23-1976-number-of-ways-to-arrive-at-destination/"
leetcode_ui: true
entry_slug: "2025-03-23-1976-number-of-ways-to-arrive-at-destination"
---

[1976. Number of Ways to Arrive at Destination](https://leetcode.com/problems/number-of-ways-to-arrive-at-destination/description/) medium
[blog post](https://leetcode.com/problems/number-of-ways-to-arrive-at-destination/solutions/6570131/kotlin-rust-by-samoylenkodmitry-lbs2/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23032025-1976-number-of-ways-to-arrive?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/9jf_tckNgiY)
![1.webp](/assets/leetcode_daily_images/42f88a14.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/936

#### Problem TLDR

Count optimal paths from 0 to n - 1 #medium #dijkstra #bellman_ford

#### Intuition

The naive Depth-First search + memoization didn't work: path is not optimal.

So, first we should establish the order of traversal, only then we can do DFS + memo.

To find the right order, we should do a Breadth-First search with priority of the minimum time. The algorithm is almost like Dijkstra in a way, that we always considering the time improvement path `time + t < ts[j]`. But to do the `counting` operation in the same loop, we should take elements strongly in the `time` increasing order.

Another way is a Bellman-Ford: we can just improve the time `n` times (but 2 is enough for the given test cases), making this algorithm O(VE) in time (or O(V + E) for the given test cases, only 2 iterations required).

#### Approach

* n <= 200, we can use bitset (c++ solution)
* for the DP, build the parents array first

#### Complexity

- Time complexity:
$$O(V + E)$$, or O(V + Elog(V)) for Dijkstra

- Space complexity:
$$O(V)$$

#### Code

```kotlin

    fun countPaths(n: Int, roads: Array<IntArray>): Int {
        val ts = LongArray(n) { Long.MAX_VALUE }; ts[0] = 0L
        val p = Array(n) { HashSet<Int>() }
        for (k in 1..2) for (r in roads) {
            if (ts[r[0]] > ts[r[1]]) r[1] = r[0].also { r[0] = r[1] }; val (a, b, t) = r
            if (ts[a] + t < ts[b]) { ts[b] = ts[a] + t; p[b] = HashSet() }
            if (ts[a] + t == ts[b]) p[b] += a
        }
        val dp = HashMap<Int, Long>(); dp[0] = 1L
        fun dfs(i: Int): Long = dp.getOrPut(i) { p[i].sumOf(::dfs) % 1_000_000_007L }
        return dfs(n - 1).toInt()
    }

```
```kotlin

    fun countPaths(n: Int, roads: Array<IntArray>): Int {
        val g = Array(n) { ArrayList<Pair<Int, Long>>() }
        for ((a, b, t) in roads) { g[a] += b to 1L * t; g[b] += a to 1L * t }
        val q = PriorityQueue<Pair<Int, Long>>(compareBy { it.second }); q += 0 to 0
        val ts = LongArray(n) { Long.MAX_VALUE }; ts[0] = 0L; val cnt = IntArray(n) { 1 }
        while (q.size > 0) q.poll().let { (i, time) ->
            for ((j, t) in g[i])
                if (time + t < ts[j]) { cnt[j] = cnt[i]; ts[j] = time + t; q += j to time + t }
                else if (time + t == ts[j]) cnt[j] = (cnt[j] + cnt[i]) % 1_000_000_007
        }
        return cnt[n - 1]
    }

```
```rust

    pub fn count_paths(n: i32, roads: Vec<Vec<i32>>) -> i32 {
        let mut g = vec![vec![]; n as usize]; for r in roads {
            let (a, b, t) = (r[0] as usize, r[1] as usize, r[2] as i64);
            g[a].push((-t, b)); g[b].push((-t, a)) }
        let (mut ts, mut cnt) = (vec![i64::MIN; g.len()], vec![1; g.len()]);
        let mut q = BinaryHeap::from_iter([(0, 0)]); ts[0] = 0;
        while let Some((time, i)) = q.pop() {
            for &(t, j) in &g[i] {
                if time + t > ts[j] { cnt[j] = cnt[i]; ts[j] = time + t; q.push((ts[j], j)) }
                else if time + t == ts[j] { cnt[j] = (cnt[j] + cnt[i]) % 1_000_000_007 }}}
        cnt[n as usize - 1]
    }

```
```c++

int countPaths(int n, vector<vector<int>>& r) {
    vector<long> ts(n, LONG_MAX), dp(n, -1); ts[0] = dp[0] = 1; vector<bitset<200>> p(n);
    for (int k = 2; k--;) for (auto& v : r) {
        if (ts[v[0]] > ts[v[1]]) swap(v[0], v[1]); int a = v[0], b = v[1], t = v[2];
        if (ts[a] + t < ts[b]) ts[b] = ts[a] + t, p[b].reset();
        if (ts[a] + t == ts[b]) p[b][a] = 1;
    }
    auto dfs = [&](this const auto& dfs, int i) -> long {
        if (dp[i] != -1) return dp[i];
        long sum = 0; for (int j = 0; j < n; ++j) if (p[i][j]) sum += dfs(j);
        return dp[i] = sum % 1'000'000'007;
    };
    return dfs(n - 1);
}

```

