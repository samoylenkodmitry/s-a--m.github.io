---
layout: leetcode-entry
title: "2582. Pass the Pillow"
permalink: "/leetcode/problem/2024-07-06-2582-pass-the-pillow/"
leetcode_ui: true
entry_slug: "2024-07-06-2582-pass-the-pillow"
---

[2582. Pass the Pillow](https://leetcode.com/problems/pass-the-pillow/description/) easy
[blog post](https://leetcode.com/problems/pass-the-pillow/solutions/5425379/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/6072024-2582-pass-the-pillow?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/CGu-DEBRZEA)
![2024-07-06_08-32_1.webp](/assets/leetcode_daily_images/982bee2f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/661

#### Problem TLDR

Loop position in increasing-decreasing array #easy #math #simulation

#### Intuition

For the interview or contest just write a simulation code, it is straghtforward: use `delta` variable and change it`s sign when `i` reaches `1` or `n`, repeat `time` times.

The O(1) solution can be derived from the observation of the repeating patterns:

```j

    //n = 4
    //t  0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21
    //i  1 2 3 4 3 2 1 2 3 4 3  2  1  2  3  4  3  2  1  2  3  4
    //     1 2 3 1 2 3 1 2 3 1  2  3  1  2  3  1  2  3  1  2  3
    //               ^
    //               t=6
    //               6/3 = 2 -> 2%2=0 (increasing)
    //               6%3 = 0 -> i=1+0
    //                 ^
    //                 t=7
    //                 7/3=2 -> 2%2=0 (increasing)
    //                 7%3=1 -> i=1+1=2
    //                     ^
    //                     t=9
    //                     9/3=3 -> 3%2=1 (decreasing)
    //                     9%3=0 -> i=4-0=4
    //                                      ^
    //                                      t=15
    //                                      15/3=5 -> 5%2=1 (decreasing)
    //                                      15%3=0 -> i=4-0=4
    //

```

There are cycles, in which `i` increases and decreases and we can say, it is `n - 1` length. From that we need to find in which kind of cycle we are and derive two cases: in increasing add remainder of cycle to `1`, in decreasing subtract the remainder from `n`.

There is another approach however, it is to consider cycle as a full round of `2 * (n - 1)` steps. Then the solution is quite similar.

#### Approach

Let's implement it first in Kotlin and second in Rust. (Simulation code I wrote on the youtube screencast, it didn't require thinking.)

#### Complexity

- Time complexity:
$$O(1)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun passThePillow(n: Int, time: Int): Int {
        val rounds = time / (n - 1)
        val rem = time % (n - 1)
        return if (rounds % 2 > 0) n - rem else 1 + rem
    }

```
```rust

    pub fn pass_the_pillow(n: i32, time: i32) -> i32 {
        let t = time % (2 * (n - 1));
        if t > n - 1 { n - (t - n) - 1 } else { t + 1 }
    }

```

