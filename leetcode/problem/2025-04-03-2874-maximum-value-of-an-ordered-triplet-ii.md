---
layout: leetcode-entry
title: "2874. Maximum Value of an Ordered Triplet II"
permalink: "/leetcode/problem/2025-04-03-2874-maximum-value-of-an-ordered-triplet-ii/"
leetcode_ui: true
entry_slug: "2025-04-03-2874-maximum-value-of-an-ordered-triplet-ii"
---

[2874. Maximum Value of an Ordered Triplet II](https://leetcode.com/problems/maximum-value-of-an-ordered-triplet-ii/description/) medium
[blog post](https://leetcode.com/problems/maximum-value-of-an-ordered-triplet-ii/solutions/6610133/kotlin-rust-by-samoylenkodmitry-mx5g/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/03042025-2874-maximum-value-of-an?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Emy7efWwTQo)
![1.webp](/assets/leetcode_daily_images/6ed301f2.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/947

#### Problem TLDR

Max (a[i] - a[j]) * a[k] #medium

#### Intuition

Same as previous (https://t.me/leetcode_daily_unstoppable/946), but more time contrained.

Compute max so-far, then max diff so-far, then max of `max diff * a[i]`.

#### Approach

* I've already golfed it yesterday
* but how about to use just a single extra variable?
* how about no extra variables?

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    var d = 0
    fun maximumTripletValue(a: IntArray) = a.maxOf { x ->
    1L * d * x.also { a[0] = max(a[0], x); d = max(d, a[0] - x) }}

```
```kotlin

    fun maximumTripletValue(a: IntArray) = (2..<a.size).maxOf { i ->
    if (i < 3) { a[1] = a[0] - a[1].also { a[0] = max(a[0], a[1]) }}
    1L * max(0, a[1]) * a[i].also {
        if (a[i] > a[0]) a[0] = a[i]; a[1] = max(a[1], a[0] - a[i]) }}

```
```rust

    pub fn maximum_triplet_value(a: Vec<i32>) -> i64 {
        a.iter().fold((0, 0, 0), |(r, d, m), &x|
        (r.max(d as i64 * x as i64), d.max(x.max(m) - x), x.max(m))).0
    }

```
```c++

    long long maximumTripletValue(vector<int>& a) {
        long long r = 0, m = 0, d = 0;
        for (int x: a) r = max(r, d * x), m = max(m, 1LL * x), d = max(d, m - x);
        return r;
    }

```

