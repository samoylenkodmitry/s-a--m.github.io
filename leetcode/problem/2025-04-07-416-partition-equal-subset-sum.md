---
layout: leetcode-entry
title: "416. Partition Equal Subset Sum"
permalink: "/leetcode/problem/2025-04-07-416-partition-equal-subset-sum/"
leetcode_ui: true
entry_slug: "2025-04-07-416-partition-equal-subset-sum"
---

[416. Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/description/) medium
[blog post](https://leetcode.com/problems/partition-equal-subset-sum/solutions/6624628/kotlin-rust-by-samoylenkodmitry-qud0/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/07042025-416-partition-equal-subset?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/6L6CMWMEsXM)
![1.webp](/assets/leetcode_daily_images/ae31c14c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/951

#### Problem TLDR

Two equal sum subsets #medium #dp

#### Intuition

This is a choice problem: either pick or skip.
The trick is to define cacheable subproblem, otherwise it is 2^n time complexity.

One accepted way is to cache possible unique subset sums from the suffix array. (slow, but accepted, O(ns), where `s` is a unique sums count).

Another way, is to set `target` and search for any subset that has this sum. O(ns).

Bottom-up is the fastest: for the current value, mark down starting from the `target - x`.

#### Approach

* The clever optimization is a bitset: each bit is a subset sum, we can mark all at once by shifting entire set by `x`

#### Complexity

- Time complexity:
$$O(ns)$$, s is up to max * (max + 1) / 2 for unique seq 1,2,3...max, so it is O(nm^2)

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun canPartition(n: IntArray, s: Int = n.sum()) = n
        .fold(setOf(0)) { s, x -> s + s.map { it + x }}
        .any { s == 2 * it }

```
```kotlin

    fun canPartition(nums: IntArray): Boolean {
        val sum = nums.sum(); var sums = setOf(0)
        for (x in nums) sums += sums.map { it + x }
        return sums.any { sum - it == it }
    }

```
```kotlin

    fun canPartition(nums: IntArray): Boolean {
        val sum = nums.sum(); val dp = HashMap<Int, Set<Int>>()
        fun dfs(i: Int): Set<Int> = if (i == nums.size) setOf(0) else
            dp.getOrPut(i) {
                dfs(i + 1) + dfs(i + 1).map { it + nums[i] }
            }
        return dfs(0).any { sum - it == it }
    }

```
```kotlin

    fun canPartition(n: IntArray): Boolean {
        n.sort()
        val sum = n.sum(); if (sum % 2 > 0) return false
        val dp = HashMap<Pair<Int, Int>, Boolean>()
        fun dfs(i: Int, t: Int): Boolean = if (i == n.size) t == 0 else
            t >= 0 && dp.getOrPut(i to t) {
                dfs(i + 1, t - n[i]) || dfs(i + 1, t)
            }
        return dfs(0, sum / 2)
    }

```
```kotlin

    fun canPartition(n: IntArray): Boolean {
        val s = n.sum(); if (s % 2 > 0) return false
        val d = IntArray(s / 2 + 1); d[0] = 1
        for (x in n) for (t in s / 2 downTo x) d[t] += d[t - x]
        return d[s / 2] != 0
    }

```
```rust

    pub fn can_partition(n: Vec<i32>) -> bool {
        let s = n.iter().sum::<i32>() as usize; if s % 2 > 0 { return false }
        let mut d = vec![0; 1 + s / 2]; d[0] = 1;
        for x in n { for t in (x as usize..=s / 2).rev() { d[t] |= d[t - x as usize]}}
        d[s / 2] > 0
    }

```
```c++

    bool canPartition(vector<int>& n) {
        int s = 0; bitset<10001> b(1);
        for (int x: n) s += x, b |= b << x;
        return (1 - s & 1) * b[s / 2];
    }

```

