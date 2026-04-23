---
layout: leetcode-entry
title: "2016. Maximum Difference Between Increasing Elements"
permalink: "/leetcode/problem/2025-06-16-2016-maximum-difference-between-increasing-elements/"
leetcode_ui: true
entry_slug: "2025-06-16-2016-maximum-difference-between-increasing-elements"
---

[2016. Maximum Difference Between Increasing Elements](https://leetcode.com/problems/maximum-difference-between-increasing-elements/description) easy
[blog post](https://leetcode.com/problems/maximum-difference-between-increasing-elements/solutions/6848831/kotlin-rust-by-samoylenkodmitry-gptu/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16062025-2016-maximum-difference?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/wLHjEigMmtI)
![1.webp](/assets/leetcode_daily_images/6623a064.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1021

#### Problem TLDR

Max increasing pair diff #easy

#### Intuition

Brute-force works.
Or, compute running min and search max(current - min).

#### Approach

* shortest code can be the optimal too

#### Complexity

- Time complexity:
$$O(n^2)$$, or O(n)

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 78ms
    fun maximumDifference(n: IntArray) =
        (0..<n.lastIndex).maxOf { i -> n.drop(i + 1).maxOf { it - n[i] }}
        .takeIf { it > 0 } ?: -1

```
```kotlin

// 37ms
    fun maximumDifference(n: IntArray) =
        (0..<n.lastIndex).maxOf { i ->
        (i + 1..<n.size).maxOf { j -> n[j] - n[i] }}
        .takeIf { it > 0 } ?: -1

```
```kotlin

// 3ms
    fun maximumDifference(n: IntArray) =
        n.fold(n[0] to 0) { (m, r), t -> min(t, m) to max(r, t - m) }
        .second.takeIf { it > 0 } ?: -1

```
```kotlin

// 1ms
    fun maximumDifference(n: IntArray): Int {
        var min = n[0]; var r = 0
        for (x in n) {
            r = max(r, x - min)
            min = min(min, x)
        }
        return if (r > 0) r else -1
    }

```
```rust

// 0ms
    pub fn maximum_difference(n: Vec<i32>) -> i32 {
        let x = n.iter().fold((0, n[0]), |(r, m), &x| (r.max(x - m), m.min(x))).0;
        if x > 0 { x } else { -1 }
    }

```
```c++

// 0ms
    int maximumDifference(vector<int>& n) {
        int r = 0, m = n[0];
        for (int x: n) r = max(r, x - m), m = min(m, x);
        return r > 0 ? r : -1;
    }

```

