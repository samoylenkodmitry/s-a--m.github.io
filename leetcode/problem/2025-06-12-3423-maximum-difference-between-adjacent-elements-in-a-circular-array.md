---
layout: leetcode-entry
title: "3423. Maximum Difference Between Adjacent Elements in a Circular Array"
permalink: "/leetcode/problem/2025-06-12-3423-maximum-difference-between-adjacent-elements-in-a-circular-array/"
leetcode_ui: true
entry_slug: "2025-06-12-3423-maximum-difference-between-adjacent-elements-in-a-circular-array"
---

[3423. Maximum Difference Between Adjacent Elements in a Circular Array](https://leetcode.com/problems/maximum-difference-between-adjacent-elements-in-a-circular-array/description/) easy
[blog post](https://leetcode.com/problems/maximum-difference-between-adjacent-elements-in-a-circular-array/solutions/6835259/kotlin-rust-by-samoylenkodmitry-d4ou/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12062025-3423-maximum-difference?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/WjkNPyza0w8)
![1.webp](/assets/leetcode_daily_images/8afbfec3.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1017

#### Problem TLDR

Max abs sibling diff #easy

#### Intuition

There are many surprising ways to write that code, try all of them.

#### Approach

* kotlin's `last()` makes runtime worse 12ms vs 1ms of `n[n.size - 1]`
* we can `windowed`
* we can `zip`
* we can zip with 0..100
* we can minimize the array reading

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 24ms
    fun maxAdjacentDistance(n: IntArray) =
        n.zip(n.drop(1) + n[0]).maxOf { (a, b) -> abs(a - b) }

```
```kotlin

// 22ms
    fun maxAdjacentDistance(n: IntArray) =
        (n + n[0]).asList().windowed(2).maxOf { abs(it[0] - it[1]) }

```
```kotlin

// 17ms
    fun maxAdjacentDistance(n: IntArray) =
        n.indices.maxOf { abs(n[(it + 1) % n.size] - n[it]) }

```
```kotlin

// 15ms
    fun maxAdjacentDistance(n: IntArray) =
        n.zip(intArrayOf(n.last()) + n).maxOf { (a, b) -> abs(a - b) }

```
```kotlin

// 11ms
    fun maxAdjacentDistance(n: IntArray): Int {
        var r = abs(n[0] - n.last())
        for (i in 1..<n.size) r = max(r, abs(n[i] - n[i - 1]))
        return r
    }

```
```kotlin

// 1ms
    fun maxAdjacentDistance(n: IntArray): Int {
        var r = abs(n[0] - n[n.size - 1])
        for (i in 1..<n.size) r = max(r, abs(n[i] - n[i - 1]))
        return r
    }

```
```rust

// 0ms
    pub fn max_adjacent_distance(n: Vec<i32>) -> i32 {
       (0..n.len()).map(|i| (n[(i + 1) % n.len()] - n[i]).abs()).max().unwrap()
    }

```
```rust

// 0ms
    pub fn max_adjacent_distance(n: Vec<i32>) -> i32 {
        (0..100).zip([&n[1..], &n[..1]].concat())
        .map(|(a, b)| (n[a] - b).abs()).max().unwrap()
    }

```
```c++

// 0ms
    int maxAdjacentDistance(vector<int>& n) {
        int r = 0, p = n[size(n) - 1];
        for (int x: n) r = max(r, abs(p - x)), p = x;
        return r;
    }

```

