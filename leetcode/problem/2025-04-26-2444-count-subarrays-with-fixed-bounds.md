---
layout: leetcode-entry
title: "2444. Count Subarrays With Fixed Bounds"
permalink: "/leetcode/problem/2025-04-26-2444-count-subarrays-with-fixed-bounds/"
leetcode_ui: true
entry_slug: "2025-04-26-2444-count-subarrays-with-fixed-bounds"
---

[2444. Count Subarrays With Fixed Bounds](https://leetcode.com/problems/count-subarrays-with-fixed-bounds/description) hard
[blog post](https://leetcode.com/problems/count-subarrays-with-fixed-bounds/solutions/6688049/kotlin-rust-by-samoylenkodmitry-qyws/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26042025-2444-count-subarrays-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/aENfuOjh1bg)
![1.webp](/assets/leetcode_daily_images/464f5969.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/970

#### Problem TLDR

Subarrays with min=minK, max=maxK #hard #two_pointers

#### Intuition

I've encounter this problem for a 3rd time (last https://leetcode.com/problems/count-subarrays-with-fixed-bounds/solutions/4951301/kotlin-rust/).

This time I felt pretty fluent with subarrays logic:
two pointers maintain the minimum valid window, and a third pointer `s` is a start position of the valid starting positions of possible subarrays `s..j`.

```j

    // 2 2 2 1 3 5 2 2 7 1 3 5    1..5
    //       j   j   i

    // 1 3 5 2 7 5      1..5
    // *minj
    //     *maxj
    // j     i
    //                  not in *range* but *equal*

```

#### Approach

* attention: subarray should have exact min=minK and max=maxK
* there is some pointer acrobatics trick: instead of a `start` position, track `count`, increase it by pointers diff

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 38ms
    fun countSubarrays(n: IntArray, minK: Int, maxK: Int): Long {
        var a = -1; var b = -1; var s = -1
        return n.withIndex().sumOf { (i, x) ->
            if (x == minK) a = i; if (x == maxK) b = i
            if (x < minK || x > maxK) { s = i; a = i; b = i }
            1L * min(a, b) - s
        }
    }

```
```kotlin
// 6ms
    fun countSubarrays(n: IntArray, minK: Int, maxK: Int): Long {
        var a = -1; var b = -1; var c = 0; var r = 0L
        for ((i, x) in n.withIndex())
            if (x < minK || x > maxK) { a = i; b = i; c = 0 }
            else {
                if (x == minK) { if (a < b) c += b - a; a = i }
                if (x == maxK) { if (b < a) c += a - b; b = i }
                r += c
            }
        return r
    }

```
```rust
// 0ms
    pub fn count_subarrays(n: Vec<i32>, min_k: i32, max_k: i32) -> i64 {
        let (mut a, mut b, mut c) = (-1, -1, 0);
        n.into_iter().enumerate().map(|(i, x)| {
            if x < min_k || x > max_k { a = i as i32; b = i as i32; c = 0 }
            if x == min_k { if a < b { c += b - a }; a = i as i32 }
            if x == max_k { if b < a { c += a - b }; b = i as i32 }
            c as i64
        }).sum()
    }

```
```c++
// 0ms
    long long countSubarrays(vector<int>& n, int minK, int maxK) {
        int a = -1, b = -1, c = 0; long long r = 0;
        for (int i = 0; i < size(n); ++i) {
            if (n[i] < minK || n[i] > maxK) a = i, b = i, c = 0;
            if (n[i] == minK) c += max(0, b - a), a = i;
            if (n[i] == maxK) c += max(0, a - b), b = i;
            r += c;
        } return r;
    }

```

