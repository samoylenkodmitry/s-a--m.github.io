---
layout: leetcode-entry
title: "2425. Bitwise XOR of All Pairings"
permalink: "/leetcode/problem/2025-01-16-2425-bitwise-xor-of-all-pairings/"
leetcode_ui: true
entry_slug: "2025-01-16-2425-bitwise-xor-of-all-pairings"
---

[2425. Bitwise XOR of All Pairings](https://leetcode.com/problems/bitwise-xor-of-all-pairings/description/) medium
[blog post](https://leetcode.com/problems/bitwise-xor-of-all-pairings/solutions/6287965/kotlin-rust-by-samoylenkodmitry-io24/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16012025-2425-bitwise-xor-of-all?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/4jtajUTdAaU)
[deep-dive](https://notebooklm.google.com/notebook/22533cbe-d611-41df-a7a5-539e71901972/audio)
![1.webp](/assets/leetcode_daily_images/7cef709a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/866

#### Problem TLDR

Xor of all pairs xors #medium #xor

#### Intuition

Observe the all pairs xor:

```j

    // 2 1 3
    // 10 2 5 0
    // 2^10 2^2 2^5 2^0
    // 1^10 1^2 1^5 1^0
    // 3^10 3^2 3^5 3^0

```

Even size will reduce other array to 0.

#### Approach

* we can use a single variable

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun xorAllNums(nums1: IntArray, nums2: IntArray) =
        nums1.reduce(Int::xor) * (nums2.size % 2) xor
        nums2.reduce(Int::xor) * (nums1.size % 2)

```
```rust

    pub fn xor_all_nums(nums1: Vec<i32>, nums2: Vec<i32>) -> i32 {
        let mut r = 0;
        if nums2.len() % 2 > 0 { for x in &nums1 { r ^= x }}
        if nums1.len() % 2 > 0 { for x in &nums2 { r ^= x }}; r
    }

```
```c++

    int xorAllNums(vector<int>& nums1, vector<int>& nums2) {
        int r = 0;
        if (nums2.size() % 2) for (int x: nums1) r ^= x;
        if (nums1.size() % 2) for (int x: nums2) r ^= x;
        return r;
    }

```

