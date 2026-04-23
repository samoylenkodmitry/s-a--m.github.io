---
layout: leetcode-entry
title: "2302. Count Subarrays With Score Less Than K"
permalink: "/leetcode/problem/2025-04-28-2302-count-subarrays-with-score-less-than-k/"
leetcode_ui: true
entry_slug: "2025-04-28-2302-count-subarrays-with-score-less-than-k"
---

[2302. Count Subarrays With Score Less Than K](https://leetcode.com/problems/count-subarrays-with-score-less-than-k/description/) hard
[blog post](https://leetcode.com/problems/count-subarrays-with-score-less-than-k/solutions/6694527/kotlin-rust-by-samoylenkodmitry-774z/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28042025-2302-count-subarrays-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/IeLBs5oPwf0)
![1.webp](/assets/leetcode_daily_images/e7e24605.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/972

#### Problem TLDR

Subarrays sum * cnt <= k #hard #two_pointers

#### Intuition

This is a standart two-pointers pattern task: always move the right pointer, move the left util condition, count how many good subarray starting point are.

#### Approach

* you can use `i - j + 1` or a separate `count` variable
* careful with `int` overflow

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 28ms
    fun countSubarrays(n: IntArray, k: Long): Long {
        var s = 0L; var j = 0
        return n.withIndex().sumOf { (i, x) ->
            s += x; while (s * (i - j + 1) >= k) s -= n[j++]
            1L + i - j
        }
    }

```
```kotlin

// 3ms
    fun countSubarrays(n: IntArray, k: Long): Long {
        var s = 0L; var r = 0L; var j = 0
        for ((i, x) in n.withIndex()) {
            s += x
            while (s * (i - j + 1) >= k) s -= n[j++]
            r += i - j + 1
        }
        return r
    }

```
```rust

// 0ms
    pub fn count_subarrays(n: Vec<i32>, k: i64) -> i64 {
        let (mut s, mut j) = (0, 0);
        n.iter().enumerate().map(|(i, &x)| {
            s += x as i64;
            while s * (i - j + 1) as i64 >= k { s -= n[j] as i64; j += 1 }
            i - j + 1
        }).sum::<usize>() as _
    }

```
```c++

// 0ms
    long long countSubarrays(vector<int>& n, long long k) {
        long long s = 0, r = 0; int j = 0, c = 0;
        for (int x: n) {
            s += x; ++c;
            while (c * s >= k) s -= n[j++], --c;
            r += c;
        } return r;
    }

```

