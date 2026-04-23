---
layout: leetcode-entry
title: "1493. Longest Subarray of 1's After Deleting One Element"
permalink: "/leetcode/problem/2025-08-24-1493-longest-subarray-of-1-s-after-deleting-one-element/"
leetcode_ui: true
entry_slug: "2025-08-24-1493-longest-subarray-of-1-s-after-deleting-one-element"
---

[1493. Longest Subarray of 1's After Deleting One Element](https://leetcode.com/problems/longest-subarray-of-1s-after-deleting-one-element/description/) medium
[blog post](https://leetcode.com/problems/longest-subarray-of-1s-after-deleting-one-element/solutions/7116389/kotlin-rust-by-samoylenkodmitry-0ony/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24082025-1493-longest-subarray-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/y6G5mYEEofQ)

![1.webp](/assets/leetcode_daily_images/e50002dc.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1091

#### Problem TLDR

Max 1-subarray after removing one elemnt #medium #two_pointers

#### Intuition

The simple way:
* count `prev` ones and `curr` ones, then max(res, prev+curr)
* corner cases are: all ones, single one island and zero

The clever way:
* use fact that we only interested in the largest island
* set left border `l` and move it always while zeros are two and more
* all the smaller islands doesn't matter

#### Approach

* the two pointers: always move right, move left until condition, compute current min/max result
* for max window sometimes we didn't have to shrink window, just move
* right border of sliding window will eventually be at `size-1`, we only interested in left

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 2ms
    fun longestSubarray(n: IntArray): Int {
        var l = 0; var z = 0
        for (x in n) z -= x - if (z > x) n[l++] else 1
        return n.size - l - 1
    }

```
```kotlin

// 4ms
    fun longestSubarray(n: IntArray): Int {
        var p = 0; var c = 0; var r = 0; var z = 1
        for (x in n) if (x > 0) r = max(r, ++c+p)
            else { p = c; c = 0; z = 0 }
        return r - z
    }

```
```rust

// 0ms
    pub fn longest_subarray(n: Vec<i32>) -> i32 {
        let (mut l, mut zs) = (0, 0);
        (0..n.len()).map(|i| {
            zs += 1 - n[i];
            if zs > 1 { zs -= 1 - n[l]; l += 1 }
            i - l
        }).max().unwrap() as _
    }

```
```c++

// 0ms
    int longestSubarray(vector<int>& n) {
        int l = 0, z = 0;
        for (int x: n) z -= x-(z>x?n[l++]:1);
        return size(n) - l - 1;
    }

```
```python

// 43ms
    def longestSubarray(_, n):
        l=z=0
        for x in n: t=z>x; z+=1-x-t+t*n[l]; l += t
        return len(n)-l-1

```

