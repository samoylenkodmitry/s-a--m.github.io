---
layout: leetcode-entry
title: "368. Largest Divisible Subset"
permalink: "/leetcode/problem/2025-04-06-368-largest-divisible-subset/"
leetcode_ui: true
entry_slug: "2025-04-06-368-largest-divisible-subset"
---

[368. Largest Divisible Subset](https://leetcode.com/problems/largest-divisible-subset/description/) medium
[blog post](https://leetcode.com/problems/largest-divisible-subset/solutions/6621195/kotlin-rust-by-samoylenkodmitry-io7v/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06042025-368-largest-divisible-subset?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/u0rALMsbjsc)
![1.webp](/assets/leetcode_daily_images/4a965cdb.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/950

#### Problem TLDR

Longest all pairs divisible subset #medium #dp

#### Intuition

Solved with a hint: dynamic programming works (and parents can be resolved later).

Previous attempt was here 1 year ago https://t.me/leetcode_daily_unstoppable/500 (where I've resorted to look at my solution)

```j

    // 1 3 4 6 8 12 16 400
    // * *   *   *
    // *   *   *    *  *
    //   2
    //     3
    //
    // 5,9,18,54,90,108,540,180,360,720
    //      *  *
    //      *     *
    // 5 3  2  2  2
    //      3  3  3
    //            5
    //        54 vs 90  choice  2^n problem

```

We should invent an example where there is an obvious chioce to be made: take or skip.

After that, look at possible memoization: if it is sorted, we can have a longest tail for every two positions and memoize it.

#### Approach

* we actually have a distinct tail for the current position, if we take previous as taken or `1` for the position `0`.
* bottom-up: always take current dp[i] and find the best prefix from dp[`0..<i`], compute parents as you go

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun largestDivisibleSubset(nums: IntArray) = buildList {
        nums.sort(); var k = 0
        val p = IntArray(nums.size) { -1 }; val dp = IntArray(nums.size)
        for (i in nums.indices)
            for (j in 0..<i) if (nums[i] % nums[j] == 0 && 1 + dp[j] > dp[i]) {
                dp[i] = 1 + dp[j]; p[i] = j; if (dp[i] > dp[k]) k = i
            }
        while (k >= 0) { add(nums[k]); k = p[k] }
    }

```
```kotlin

    fun largestDivisibleSubset(nums: IntArray): List<Int> {
        nums.sort(); val dp = HashMap<Int, List<Int>>()
        fun dfs(i: Int): List<Int> = dp.getOrPut(i) {
            var x = if (i == 0) 1 else nums[i - 1]
            var max = listOf<Int>()
            for (j in i..<nums.size) if (nums[j] % x == 0) {
                val next = listOf(nums[j]) + dfs(j + 1)
                if (next.size > max.size) max = next
            }
            max
        }
        return dfs(0)
    }

```
```kotlin

    fun largestDivisibleSubset(nums: IntArray): List<Int> {
        nums.sort()
        val next = IntArray(nums.size + 1) { -1 }
        val dp = HashMap<Pair<Int, Int>, Int>()
        fun dfs(i: Int, k: Int): Int = dp.getOrPut(i to k) {
            if (i == nums.size) return 0
            val skip = dfs(i + 1, k)
            val take = if (k == nums.size || nums[i] % nums[k] == 0) 1 + dfs(i + 1, i) else 0
            if (take > skip) next[k] = i
            max(skip, take)
        }
        dfs(0, nums.size)
        var i = nums.size; val res = mutableListOf<Int>()
        while (next[i] >= 0) { res += nums[next[i]]; i = next[i] }
        return res
    }

```
```rust

    pub fn largest_divisible_subset(mut n: Vec<i32>) -> Vec<i32> {
        n.sort_unstable(); let (mut k, mut r) = (0, vec![]);
        let (mut p, mut d) = (vec![n.len(); n.len()], vec![0; n.len()]);
        for i in 0..n.len() { for j in 0..i {
            if n[i] % n[j] == 0 && 1 + d[j] > d[i] {
                d[i] = 1 + d[j]; p[i] = j
            }} if d[i] > d[k] { k = i } }
        while k < n.len() { r.push(n[k]); k = p[k] }; r
    }

```
```c++

    vector<int> largestDivisibleSubset(vector<int>& n) {
        sort(begin(n), end(n));
        vector<int> r, p(size(n), -1), d(size(n)); int k = 0;
        for (int i = 0; i < size(n); ++i) for (int j = 0; j < i; ++j) {
            if (n[i] % n[j] == 0 && 1 + d[j] > d[i]) d[i] = 1 + d[j], p[i] = j;
            if (d[i] > d[k]) k = i; }
        while (k >= 0) { r.push_back(n[k]); k = p[k]; }
        return r;
    }

```

