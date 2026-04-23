---
layout: leetcode-entry
title: "2962. Count Subarrays Where Max Element Appears at Least K Times"
permalink: "/leetcode/problem/2025-04-29-2962-count-subarrays-where-max-element-appears-at-least-k-times/"
leetcode_ui: true
entry_slug: "2025-04-29-2962-count-subarrays-where-max-element-appears-at-least-k-times"
---

[2962. Count Subarrays Where Max Element Appears at Least K Times](https://leetcode.com/problems/count-subarrays-where-max-element-appears-at-least-k-times/description) medium
[blog post](https://leetcode.com/problems/count-subarrays-where-max-element-appears-at-least-k-times/solutions/6697812/kotlin-rust-by-samoylenkodmitry-xa55/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29042025-2962-count-subarrays-where?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ZJr4KICz5c4)
![1.webp](/assets/leetcode_daily_images/3c5ee217.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/973

#### Problem TLDR

Subarrays at least k maxes #medium #two_pointers

#### Intuition

Two pointers pattern:
* always move the right
* move the left until condition
* count how many valid subarray starting positions are

```j
    // 1,3,2,3,3
    //   j   i i
```

#### Approach

* we can compute `max` as we go
* we can use a queue instead of the second pointer (slower runtime though)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$, or O(n) or O(k) for the queue solution

#### Code

```kotlin

// 43ms
    fun countSubarrays(n: IntArray, k: Int): Long {
        var m = n.max(); val q = ArrayList<Int>()
        return n.withIndex().sumOf { (i, x) ->
            if (x == m) q += i
            if (q.size >= k) 1L + q[q.size - k] else 0L
        }
    }

```
```kotlin

// 20ms
    fun countSubarrays(n: IntArray, k: Int): Long {
        var j = 0; var m = n.max(); var c = 0
        return n.sumOf { x ->
            if (x == m) ++c
            while (c >= k) if (n[j++] == m) --c
            1L * j
        }
    }

```
```kotlin

// 13ms
    fun countSubarrays(n: IntArray, k: Int): Long {
        var r = 0L; var m = 0; val q = ArrayList<Int>()
        for (i in n.indices) {
            if (n[i] > m) { r = 0; m = n[i]; q.clear(); q += i }
            else if (n[i] == m) q += i
            if (q.size >= k) r += q[q.size - k] + 1
        }
        return r
    }

```
```kotlin

// 8ms
    fun countSubarrays(n: IntArray, k: Int): Long {
        var j = 0; var r = 0L; var m = 0; var c = 0
        for (x in n) {
            if (x > m) { r = 0; c = 1; j = 0; m = x } else if (x == m) ++c
            while (c >= k) if (n[j++] == m) --c
            r += j
        }
        return r
    }

```
```kotlin

// 6ms
    fun countSubarrays(n: IntArray, k: Int): Long {
        var j = 0; var r = 0L; var m = 0; var c = 0
        for ((i, x) in n.withIndex()) {
            if (x > m) { r = 0; c = 1; j = i; m = x } else if (x == m) ++c
            while (c > k || n[j] != m) if (n[j++] == m) --c
            if (c == k) r += j + 1
        }
        return r
    }

```
```rust

// 0ms
    pub fn count_subarrays(n: Vec<i32>, k: i32) -> i64 {
        let (mut j, mut r, mut m, mut c) = (0, 0, 0, 0);
        for &x in &n {
            if x > m { r = 0; c = 1; j = 0; m = x } else if x == m { c += 1 }
            while c >= k { if n[j] == m { c -= 1 }; j += 1 }
            r += j as i64
        } r
    }

```
```c++

// 0ms
    long long countSubarrays(vector<int>& n, int k) {
        int j = 0, m = 0, c = 0; long long r = 0;
        for (int x: n) {
            if (x > m) r = 0, c = 1, j = 0, m = x; else if (x == m) ++c;
            while (c >= k) if (n[j++] == m) --c;
            r += j;
        } return r;
    }

```

