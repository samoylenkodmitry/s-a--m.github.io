---
layout: leetcode-entry
title: "2873. Maximum Value of an Ordered Triplet I"
permalink: "/leetcode/problem/2025-04-02-2873-maximum-value-of-an-ordered-triplet-i/"
leetcode_ui: true
entry_slug: "2025-04-02-2873-maximum-value-of-an-ordered-triplet-i"
---

[2873. Maximum Value of an Ordered Triplet I](https://leetcode.com/problems/maximum-value-of-an-ordered-triplet-i/description/) easy
[blog post](https://leetcode.com/problems/maximum-value-of-an-ordered-triplet-i/solutions/6606186/kotlin-rust-by-samoylenkodmitry-hsx8/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/02042025-2873-maximum-value-of-an?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/b5Xbg4k-q1s)
![1.webp](/assets/leetcode_daily_images/9e4aa2a8.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/946

#### Problem TLDR

Max (a[i] - a[j]) * a[k] #easy

#### Intuition

Brute-force works.
But can we do better?
* heap solution: track max so-far, diff with current, take max from the right by polling from heap, skipping visited indices
* linear solution: track max so-far, track max diff, multiply max diff and the current value

#### Approach

* carefult to not overlap indices

#### Complexity

- Time complexity:
$$O(n)$$, or n^3 for brute-force, or nlog(n) for heap

- Space complexity:
$$O(1)$$, or O(n) for heap

#### Code

```kotlin

    fun maximumTripletValue(n: IntArray): Long {
        var m = 0L; var d = 0L
        return n.maxOf { x ->
            val r = d * x; d = max(d, m - x); m = max(1L * x, m); r
        }
    }

```
```kotlin

    fun maximumTripletValue(n: IntArray) = n.indices.maxOf { i ->
        (i + 1..<n.size).maxOfOrNull { j ->
        (j + 1..<n.size).maxOfOrNull { k ->
        (1L * n[i] - n[j]) * n[k] } ?: 0L } ?: 0L }

```
```kotlin

    fun maximumTripletValue(n: IntArray): Long {
        val q = PriorityQueue<Int>(compareBy { -n[it] })
        var r = 0L; var max = 0L; for (i in n.indices) q += i
        for (i in n.indices) {
            while (q.size > 0 && q.peek() <= i) q.poll()
            if (q.size > 0 && (max - n[i] > 0)) r = max(r, (max - n[i]) * n[q.peek()])
            max = max(1L * n[i], max)
        }
        return r
    }

```
```rust

    pub fn maximum_triplet_value(n: Vec<i32>) -> i64 {
        let (mut m, mut d) = (0, 0);
        n.iter().map(|&x| { let x = x as i64;
            let r = d * x; d = d.max(m - x); m = m.max(x); r
        }).max().unwrap()
    }

```
```c++

    long long maximumTripletValue(vector<int>& n) {
        long long m = 0, d = 0, r = 0;
        for (int x: n) r = max(r, d * x), d = max(d, m - x), m = max(m, 1LL * x);
        return r;
    }

```

