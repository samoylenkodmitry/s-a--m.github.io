---
layout: leetcode-entry
title: "3191. Minimum Operations to Make Binary Array Elements Equal to One I"
permalink: "/leetcode/problem/2025-03-19-3191-minimum-operations-to-make-binary-array-elements-equal-to-one-i/"
leetcode_ui: true
entry_slug: "2025-03-19-3191-minimum-operations-to-make-binary-array-elements-equal-to-one-i"
---

[3191. Minimum Operations to Make Binary Array Elements Equal to One I](https://leetcode.com/problems/minimum-operations-to-make-binary-array-elements-equal-to-one-i/description/) medium
[blog post](https://leetcode.com/problems/minimum-operations-to-make-binary-array-elements-equal-to-one-i/solutions/6554321/kotlin-rust-by-samoylenkodmitry-vvo7/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19032025-3191-minimum-operations?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/QdUq8hMeTzU)
![1.webp](/assets/leetcode_daily_images/e76adaf3.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/932

#### Problem TLDR

3-bits flips to make all 1 #medium #bit_manipulation

#### Intuition

The order of operations is irrelevant, flip greedily from start to finish.

#### Approach

* we can modify in-place
* or we can use bitmask with 3 bits, to flip it xor with `7 = b111`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun minOperations(nums: IntArray) =
        nums.indices.count { i ->
            (nums[i] < 1).also {
            if (it) for (j in i..i + 2) nums[j] =
                1 - (nums.getOrNull(j) ?: return -1) }
        }

```
```rust

    pub fn min_operations(nums: Vec<i32>) -> i32 {
        let (mut r, mut b) = (0, 0);
        for i in 0..nums.len() {
            b ^= nums[i]; r += b & 1 ^ 1; b ^= 7 * (b & 1 ^ 1); b >>= 1
        } if b == 0 { r } else { -1 }
    }

```
```c++

    int minOperations(vector<int>& n) {
        int r = 0;
        for (int i = 0; i < size(n); ++i) {
            if (n[i]) continue; r++;
            if (i + 3 > size(n)) return -1;
            n[i + 1] ^= 1; n[i + 2] ^= 1;
        } return r;
    }

```

