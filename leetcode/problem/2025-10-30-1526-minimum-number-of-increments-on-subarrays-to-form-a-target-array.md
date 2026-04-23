---
layout: leetcode-entry
title: "1526. Minimum Number of Increments on Subarrays to Form a Target Array"
permalink: "/leetcode/problem/2025-10-30-1526-minimum-number-of-increments-on-subarrays-to-form-a-target-array/"
leetcode_ui: true
entry_slug: "2025-10-30-1526-minimum-number-of-increments-on-subarrays-to-form-a-target-array"
---

[1526. Minimum Number of Increments on Subarrays to Form a Target Array](https://leetcode.com/problems/minimum-number-of-increments-on-subarrays-to-form-a-target-array/description) hard
[blog post](https://leetcode.com/problems/minimum-number-of-increments-on-subarrays-to-form-a-target-array/solutions/7312982/kotlin-rust-by-samoylenkodmitry-2wvy/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/30102025-1526-minimum-number-of-increments?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/RPzKFFB1HmA)

![b16d3eb3-4cea-4656-acb3-37d6e6d4dbc8 (1).webp](/assets/leetcode_daily_images/c0468ce7.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1158

#### Problem TLDR

Min +1 range increases to make target from zeros #hard

#### Intuition

```j
    // *********  1
    // **  *****  2, 3
    // **  ** **  4, 5, 6
    //     *      7
    // 331143233

    // 6 7 7 2 1 2 2 1 3
    // 6                   6 levels
    //   7                 +1 level, (6..1 levels continue)
    //     7               +0 levels, (7..1 levels continue)
    //       2             -5 levels (7..3 stop, 2..1 continue) +5 ops
    //         1           -1        (2..2 stop, 1..1 continue) +1 ops
    //           2         +1        (1..2 continue)
    //             2       +0
    //               1     -1        (2..2 stop, 1..1 continue) +1 ops
    //                 3   +2        1..3 continue
    //                  end          (1..3 stop) +2 ops and +1 for level 1
```

Optimal strategy is the Tetris game: remove islands from the bottom.
The number of ops is the number of islands.
The number of islands is the number of decreases.

#### Approach

* draw the picture
* assume algorithm is linear, try to gain as much information at each step as possible

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 58ms
    fun minNumberOperations(t: IntArray) =
        t.zip(t.drop(1)+0).sumOf { (a,b) -> max(0, a-b)}

```
```rust
// 0ms
    pub fn min_number_operations(t: Vec<i32>) -> i32 {
        t[0] + t.windows(2).map(|w| 0.max(w[1]-w[0])).sum::<i32>()
    }

```

