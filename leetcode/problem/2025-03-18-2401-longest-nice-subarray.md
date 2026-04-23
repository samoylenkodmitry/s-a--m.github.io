---
layout: leetcode-entry
title: "2401. Longest Nice Subarray"
permalink: "/leetcode/problem/2025-03-18-2401-longest-nice-subarray/"
leetcode_ui: true
entry_slug: "2025-03-18-2401-longest-nice-subarray"
---

[2401. Longest Nice Subarray](https://leetcode.com/problems/longest-nice-subarray/description/) medium
[blog post](https://leetcode.com/problems/longest-nice-subarray/solutions/6550425/kotlin-rust-by-samoylenkodmitry-w3nu/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18032025-2401-longest-nice-subarray?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/r1FzWE06ZyM)
![1.webp](/assets/leetcode_daily_images/4d9d5965.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/931

#### Problem TLDR

Max window of all pairs AND zero #medium #two_pointers #bit_manipulation

#### Intuition

All pairs AND would be zero only if they didn't share any common bits.
We can use bits counter and use two pointers: always move the right, move the left until all bits count is no more than 1.

#### Approach

* we can just use a mask: keep window valid, then we can remove number by xoring it
* the hint is: max window size is always 32, we can golf with it

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun longestNiceSubarray(nums: IntArray) = (32 downTo 1).first {
        nums.asList().windowed(it).any { w ->
            (0..32).all { b -> w.sumOf { it shr b and 1 } < 2 }}
    }

```
```kotlin

    fun longestNiceSubarray(nums: IntArray): Int {
        var u = 0; var j = 0
        return nums.indices.maxOf { i ->
            while (u and nums[i] > 0) u = u xor nums[j++]
            u = u or nums[i]; i - j + 1
        }
    }

```
```rust

    pub fn longest_nice_subarray(nums: Vec<i32>) -> i32 {
        let (mut u, mut j) = (0, 0);
        (0..nums.len()).map(|i| {
            while u & nums[i] > 0 { u ^= nums[j]; j += 1 }
            u |= nums[i]; i - j + 1
        }).max().unwrap() as _
    }

```
```c++

    int longestNiceSubarray(vector<int>& n) {
        int u = 0, j = 0, r = 0;
        for (int i = 0; i < size(n); u |= n[i++]) {
            while (u & n[i]) u ^= n[j++]; r = max(r, i - j + 1);
        } return r;
    }

```

