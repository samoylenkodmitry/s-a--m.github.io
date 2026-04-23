---
layout: leetcode-entry
title: "1863. Sum of All Subset XOR Totals"
permalink: "/leetcode/problem/2025-04-05-1863-sum-of-all-subset-xor-totals/"
leetcode_ui: true
entry_slug: "2025-04-05-1863-sum-of-all-subset-xor-totals"
---

[1863. Sum of All Subset XOR Totals](https://leetcode.com/problems/sum-of-all-subset-xor-totals/description) easy
[blog post](https://leetcode.com/problems/sum-of-all-subset-xor-totals/solutions/6617371/kotlin-rust-by-samoylenkodmitry-go7s/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05042025-1863-sum-of-all-subset-xor?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/7YJAoVjff5M)
![1.webp](/assets/leetcode_daily_images/93d8a012.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/949

#### Problem TLDR

Sum of all subsets xors #easy #bit #math

#### Intuition

There are total `2^(n - 1)` subsets. We can
* iterate over mask of set bits, each bit is a position
* use recursion with backtracking: take value or skip, compute `xor` so far

The math trick solution: if bit is present it contributes `2^(n - 1)` times, find all present bits with `or`.

#### Approach

* the mask solution is easier to not make a mistake

#### Complexity

- Time complexity:
$$O(2^n)$$

- Space complexity:
$$O(n)$$, or O(1) for iterative mask solution

#### Code

```kotlin

    fun subsetXORSum(a: IntArray) = (0..(1 shl a.size)).sumOf { m ->
        a.indices.map { a[it] * (m shr it and 1) }.reduce(Int::xor)
    }

```
```kotlin

    fun subsetXORSum(a: IntArray) = (0..(1 shl a.size)).sumOf { m ->
        1 * a.indices.fold(0) { x, i -> x xor a[i] * (m shr i and 1) }
    }

```
```kotlin

    fun subsetXORSum(a: IntArray): Int {
        fun d(i: Int, x: Int): Int = if (i == a.size) x
            else d(i + 1, x) + d(i + 1, x xor a[i])
        return d(0, 0)
    }

```
```rust

    pub fn subset_xor_sum(a: Vec<i32>) -> i32 {
        (0..1 << a.len()).map(|m| (0..a.len())
        .fold(0, |x, i| x ^ a[i] * (m >> i & 1))).sum()
    }

```
```rust

    pub fn subset_xor_sum(a: Vec<i32>) -> i32 {
        fn d(a: &[i32], x: i32) -> i32
            { if a.len() > 0  { d(&a[1..], x) + d(&a[1..], x ^ a[0]) } else { x }}
        d(&a, 0)
    }

```
```c++

    int subsetXORSum(vector<int>& a) {
        for (int x: a) a[0] |= x; return a[0] << (size(a) - 1);
    }

```

