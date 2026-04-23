---
layout: leetcode-entry
title: "1509. Minimum Difference Between Largest and Smallest Value in Three Moves"
permalink: "/leetcode/problem/2024-07-03-1509-minimum-difference-between-largest-and-smallest-value-in-three-moves/"
leetcode_ui: true
entry_slug: "2024-07-03-1509-minimum-difference-between-largest-and-smallest-value-in-three-moves"
---

[1509. Minimum Difference Between Largest and Smallest Value in Three Moves](https://leetcode.com/problems/minimum-difference-between-largest-and-smallest-value-in-three-moves/description/) medium
[blog post](https://leetcode.com/problems/minimum-difference-between-largest-and-smallest-value-in-three-moves/solutions/5406773/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/3072024-1509-minimum-difference-between?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/A3v-txSOmuQ)
![2024-07-03_07-33_1.webp](/assets/leetcode_daily_images/f425a138.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/658

#### Problem TLDR

Min difference after 3 changes in array #medium #sliding_window #dynamic_programming

#### Intuition

Let's observe some examples and try to derive the algorithm:

```j
    //  1  3  5  7  9  11
    //  min = 1
    //  max4 = (5, 7, 9, 11)
    //  res = 5 - 1 = 4
    //
    //  0 1 1 4 6 6 6
    //  min = 0
    //  max4 = 4 6 6 6
    //
    //  20 75 81 82 95
    //  55          13
    //      i
    //      6
    //           j
    //           1
    //         i
```
As we see, we cannot just take top 3 max or top 3 min, there are corner cases, where some min and some max must be taken. So, we can do a full search of 3 boolean choices, 2^3 total comparisons in a Depth-First search manner.
Another way to look at the problem as suffix-prefix trimming:
0 prefix + 3 suffix, 1 prefix + 2 suffix, 2 prefix + 1 suffix, 3 prefix + 0 suffix. So, a total of 4 comparisons in a Sliding Window manner.

#### Approach

Let's implement both approaches.

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun minDifference(nums: IntArray): Int {
        nums.sort()
        fun dfs(i: Int, j: Int, n: Int): Int =
            if (i == j) 0 else if (i > j) Int.MAX_VALUE
            else if (n > 2) nums[j] - nums[i]
            else min(dfs(i + 1, j, n + 1), dfs(i, j - 1, n + 1))
        return dfs(0, nums.lastIndex, 0)
    }

```
```rust

    pub fn min_difference(mut nums: Vec<i32>) -> i32 {
        let n = nums.len(); if n < 4 { return 0 }; nums.sort_unstable();
        (0..4).map(|i| nums[n - 4 + i] - nums[i]).min().unwrap()
    }

```

