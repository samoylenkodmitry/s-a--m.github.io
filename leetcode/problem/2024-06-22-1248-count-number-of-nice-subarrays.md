---
layout: leetcode-entry
title: "1248. Count Number of Nice Subarrays"
permalink: "/leetcode/problem/2024-06-22-1248-count-number-of-nice-subarrays/"
leetcode_ui: true
entry_slug: "2024-06-22-1248-count-number-of-nice-subarrays"
---

[1248. Count Number of Nice Subarrays](https://leetcode.com/problems/count-number-of-nice-subarrays/description/) medium
[blog post](https://leetcode.com/problems/count-number-of-nice-subarrays/solutions/5349876/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22062024-1248-count-number-of-nice?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/oMp7BfPNCkg)
![2024-06-22_07-18_1.webp](/assets/leetcode_daily_images/9050df6f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/647

#### Problem TLDR

Count subarrays with `k` odds #medium #sliding_window

#### Intuition

Let's observe the problem:

```j
    // 1 1 2 1 1      k=3
    // * * * *
    //   * * * *

    // 0 1 2 3 4 5 6 7 8 9
    // 2 2 2 1 2 2 1 2 2 2  k=2
    //           .          count
    // i         .          0
    //   i       .          0
    //     i     .          0
    //       i   .          1 < k
    //         i .
    //           i
    //             i        2 == k, +4 [0..6],[1..6],[2..6],[3..6]
    //               i      2 == k  +4  0..7 1..7 2..7 3..7
    //                 i    2 == k  +4  0..8 1..8 2..8 3..8
    //                   i  2 == k  +4  0..9 1..9 2..9 3..9
```

When we find a good window `[3..6]` we must somehow calculate the number of contiguous subarrays. Let's experiment how we can do it in a single pass: when i = 6 we must add to the result all subarrays `0..6 1..6 2..6 3..6` and stop until the first `odd`. So, let's use a third pointer `border` to count the number of prefix subarrays: `j - border`.

#### Approach

* Using `sumOf` can shorten some lines of code.
* `& 1` gives `1` for `odd` numbers.
* Some conditions are exclusive to each other, and we can skip them: `cnt > 0` means `j` will stop at least once. (Don't do this in an interview, just use `j < nums.len()`.)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun numberOfSubarrays(nums: IntArray, k: Int): Int {
        var border = -1; var j = 0; var cnt = 0
        return nums.sumOf { n ->
            cnt += n and 1
            while (cnt > k) {
                border = j
                cnt -= nums[j++] and 1
            }
            while (cnt > 0 && nums[j] % 2 < 1) j++
            if (cnt < k) 0 else j - border
        }
    }

```
```rust

    pub fn number_of_subarrays(nums: Vec<i32>, k: i32) -> i32 {
        let (mut b, mut cnt, mut j) = (-1, 0, 0);
        nums.iter().map(|n| {
            cnt += n & 1;
            while cnt > k { b = j as i32; cnt -= nums[j] & 1; j += 1 }
            while cnt > 0 && nums[j] & 1 < 1 { j += 1 }
            if cnt < k { 0 } else { j as i32 - b }
        }).sum()
    }

```

