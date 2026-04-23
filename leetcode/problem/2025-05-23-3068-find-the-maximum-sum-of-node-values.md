---
layout: leetcode-entry
title: "3068. Find the Maximum Sum of Node Values"
permalink: "/leetcode/problem/2025-05-23-3068-find-the-maximum-sum-of-node-values/"
leetcode_ui: true
entry_slug: "2025-05-23-3068-find-the-maximum-sum-of-node-values"
---

[3068. Find the Maximum Sum of Node Values](https://leetcode.com/problems/find-the-maximum-sum-of-node-values/description/) hard
[blog post](https://leetcode.com/problems/find-the-maximum-sum-of-node-values/solutions/6772330/kotlin-rust-by-samoylenkodmitry-c2fw/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23052025-3068-find-the-maximum-sum?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/XrMRF6DmDrc)
![1.webp](/assets/leetcode_daily_images/2da10f8f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/997

#### Problem TLDR

Max sum by k-xoring edges #hard #tree #dp

#### Intuition

Didn't solved the second time (2024 was the previous attempt)

```j

    // 1 - 0 - 2
    // a - b - c
    // ^k  ^k
    //     ^k  ^k   a^k - b - c^k
    // a - b - c - d
    // ^   ^
    //     ^   ^
    //         ^   ^   a^k - b - c - d^k
    //
    // a^k b   c   d^k
    // a^k b   c^k d
    // a^k b^k c   d
    // a^k b^k c^k d^k
    // a   b^k c^k d
    // a   b   c^k d^k
    // a   b^k c   d^k

    // a - b - c    a*- b*- c - e*
    //     |            |
    //     d            d*
    // didn't see any simple law
    // maybe full search?
    // wong answer: careful with flipping the last (it flips the previous?)
    // 0-2-4-3
    //   |
    //   1
    //
    // 5-0-1*-3*-6-2*
    //     |
    //     4*
    // flip current without flipping previous:
    // 1. if has next
    // 51 minutes, use hints, looks like the same dp (and what is parity?)
    // looks like i did the same mistake in 2024 (and didn't finished dp)

```

What was missing:
* I didn't paid attention to the detail: *only even number of flipped numbers is possible*

Why my DFS+cache simultaion didn't worked:
* when children count > 1, we can't flip them all simultaneously

#### Approach

* attention to details: how many flips can be done, how flips happen when node has many chilren
* can you rewrite DFS dp to return a single Long result?

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$, O(n) for dp

#### Code

```kotlin

    fun maximumValueSum(nums: IntArray, k: Int, edges: Array<IntArray>): Long {
        var sum = 0L; var xored = 0; var diff = Int.MAX_VALUE / 2
        for (x in nums) {
            sum += 1L * max(x, x xor k)
            if (x xor k > x) xored++
            diff = min(diff, abs((x xor k) - x))
        }
        return sum - diff * (xored % 2)
    }

```
```kotlin

    fun maximumValueSum(nums: IntArray, k: Int, edges: Array<IntArray>): Long {
        val g = Array(nums.size) { ArrayList<Int>() }
        for ((u, v) in edges) { g[u] += v; g[v] += u }
        val dp = HashMap<Int, Pair<Long, Long>>()
        fun dfs(u: Int, p: Int): Pair<Long, Long> = dp.getOrPut(u) {
            var sumFlip = Long.MIN_VALUE / 2
            var sumStay = 0L
            for (v in g[u]) if (v != p) {
                val (flip, stay) = dfs(v, u)
                sumFlip = max(sumStay + stay, sumFlip + flip).also {
                sumStay = max(sumStay + flip, sumFlip + stay) }
            }
            val stay = nums[u]
            val flip = stay xor k
            sumFlip = max(sumStay + stay, sumFlip + flip).also {
            sumStay = max(sumStay + flip, sumFlip + stay) }
            sumFlip to sumStay
        }
        return dfs(0, -1).first
    }

```
```kotlin

    fun maximumValueSum(nums: IntArray, k: Int, edges: Array<IntArray>): Long {
        val g = Array(nums.size) { ArrayList<Int>() }
        for ((u, v) in edges) { g[u] += v; g[v] += u }
        val dp = HashMap<Pair<Int, Int>, Long>()
        fun dfs(u: Int, p: Int, f: Int): Long = dp.getOrPut(u to f) {
            val flip = (nums[u] xor k xor f).toLong()
            val stay = (nums[u] xor f).toLong()
            var sum = max(flip, stay)
            var flips = if (flip > stay) 1 else 0
            var diff = abs(flip - stay)
            for (v in g[u]) if (v != p) {
                val flip = dfs(v, u, k)
                val stay = dfs(v, u, 0)
                if (flip > stay) flips = flips xor 1
                diff = min(diff, abs(flip - stay))
                sum += max(flip, stay)
            }
            sum - diff * flips
        }
        return dfs(0, -1, 0)
    }

```
```rust

    pub fn maximum_value_sum(n: Vec<i32>, k: i32, edges: Vec<Vec<i32>>) -> i64 {
        let (mut s, mut c, mut d) = (0, 0, i32::MAX);
        for x in n {
            s += x.max(x ^ k) as i64;
            if x ^ k > x { c ^= 1 }
            d = d.min(((x ^ k) - x).abs())
        } s - d as i64 * c
    }

```
```c++

    long long maximumValueSum(vector<int>& n, int k, vector<vector<int>>& edges) {
        long long s = 0, c = 0, d = 1e9;
        for (int& x: n) {
            s += max(x, x ^ k);
            if ((x ^ k) > x) c ^= 1;
            d = min(d, 1LL * abs((x ^ k) - x));
        } return s - d * c;
    }

```

