---
layout: leetcode-entry
title: "995. Minimum Number of K Consecutive Bit Flips"
permalink: "/leetcode/problem/2024-06-24-995-minimum-number-of-k-consecutive-bit-flips/"
leetcode_ui: true
entry_slug: "2024-06-24-995-minimum-number-of-k-consecutive-bit-flips"
---

[995. Minimum Number of K Consecutive Bit Flips](https://leetcode.com/problems/minimum-number-of-k-consecutive-bit-flips/description/) medium
[blog post](https://leetcode.com/problems/minimum-number-of-k-consecutive-bit-flips/solutions/5359962/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24062024-995-minimum-number-of-k?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Mxt2xfKHDTg)
![2024-06-24_07-04_1.webp](/assets/leetcode_daily_images/d4e07fcd.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/649

#### Problem TLDR

Count `k`-range flips in binary array to make all `1` #hard #sliding_window

#### Intuition

We should flip all the `0`, so let's do it greedily. The hardness of the problem lies in the question of how much flips are already done for the current position. Let's observe an example:

```j

    // 0 1 2 3 4 5 6 7   k=3
    // 0 0 0 1 0 1 1 0  flip
    // * * *            [0..2]
    //         * * *    [4..6]
    //           * * *  [5..7]
    //           ^ how much flips in 3..5 range
    //                            or >= 3

```

If we maintain a window of `i-k+1..i`, we shall remember only the flips in this window and can safely drop all the flips in `0..i-k` range.

#### Approach

The greedy is hard to prove, so try as much examples as possible.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun minKBitFlips(nums: IntArray, k: Int): Int {
        var total = 0; var flips = ArrayDeque<Int>()
        for ((i, n) in nums.withIndex()) {
            while (flips.size > 0 && flips.first() + k < i + 1)
                flips.removeFirst()
            if ((1 - n + flips.size) % 2 > 0) {
                total++
                flips += i
                if (i + k > nums.size) return -1
            }
        }
        return total
    }

```
```rust

    pub fn min_k_bit_flips(nums: Vec<i32>, k: i32) -> i32 {
        let (mut total, mut flips) = (0, VecDeque::new());
        for (i, n) in nums.iter().enumerate() {
            while flips.front().unwrap_or(&i) + (k as usize) < i + 1
                { flips.pop_front(); }
            if (1 - n  + flips.len() as i32) % 2 > 0 {
                total += 1;
                if i + k as usize > nums.len() { return -1 }
                flips.push_back(i)
            }
        }; total as i32
    }

```

