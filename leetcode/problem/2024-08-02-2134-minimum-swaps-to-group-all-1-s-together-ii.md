---
layout: leetcode-entry
title: "2134. Minimum Swaps to Group All 1's Together II"
permalink: "/leetcode/problem/2024-08-02-2134-minimum-swaps-to-group-all-1-s-together-ii/"
leetcode_ui: true
entry_slug: "2024-08-02-2134-minimum-swaps-to-group-all-1-s-together-ii"
---

[2134. Minimum Swaps to Group All 1's Together II](https://leetcode.com/problems/minimum-swaps-to-group-all-1s-together-ii/description/) medium
[blog post](https://leetcode.com/problems/minimum-swaps-to-group-all-1s-together-ii/solutions/5572682/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/02082024-2134-minimum-swaps-to-group?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ahatWzvgMKw)
![2024-08-02_08-58_1.webp](/assets/leetcode_daily_images/cee9bb8d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/690

#### Problem TLDR

Min swaps to make a circular [01] array #medium #sliding_window

#### Intuition

I've used the first hint: consider what the final array would look like.
Let's explore all the possible arrays:

```j

    // 0123456789
    // 0010011100
    //
    // 1111000000 -> 3
    // 0111100000 -> 3
    // 0011110000 -> 3
    // 0001111000 -> 2
    // 0000111100 -> 1
    // 0000011110 -> 1
    // 0000001111 -> 2
    // 1000000111 -> 3
    // 1100000011 -> 4
    // 1110000001 -> 3

```
As we compute the necessary swaps count for each array the intuition forms: count the mismatched values, or `xor`'s.

We can use a sliding window technique to code this.

#### Approach

* use `sum` to count `1`s
* `1-nums[i]` will count `0`s
* simplify the math formula to look cool (don't do it in a real project)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun minSwaps(nums: IntArray): Int {
        val c1 = nums.sum()
        var miss = (0..<c1).count { nums[it] < 1 }
        return nums.indices.minOf { i ->
            miss += nums[i] - nums[(i + c1) % nums.size]
            miss
        }
    }

```
```rust

    pub fn min_swaps(nums: Vec<i32>) -> i32 {
        let c1 = nums.iter().sum::<i32>() as _;
        let mut miss = (0..c1).map(|i| 1 - nums[i]).sum();
        (0..nums.len()).map(|i| {
            miss += nums[i] - nums[(i + c1) % nums.len()];
            miss
        }).min().unwrap()
    }

```

