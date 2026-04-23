---
layout: leetcode-entry
title: "279. Perfect Squares"
permalink: "/leetcode/problem/2024-02-08-279-perfect-squares/"
leetcode_ui: true
entry_slug: "2024-02-08-279-perfect-squares"
---

[279. Perfect Squares](https://leetcode.com/problems/perfect-squares/description) medium
[blog post](https://leetcode.com/problems/perfect-squares/solutions/4695798/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/08022024-279-perfect-squares?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/3niMLL5clIo)
![image.png](/assets/leetcode_daily_images/06819090.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/499

#### Problem TLDR

Min square numbers sum up to `n`.

#### Intuition

By wrong intuition would be just subtract maximum possible square number: 12 = 9 + _remainder_. So, we should explore all of possible squares and choose min count of them. We can do DFS and cache the result. To pass the TLE, we need to rewrite it back into bottom up DP.

#### Approach

Let's write as shorter as we can by using:
* Kotlin: `minOf`, `sqrt` without `Math`, `toFloat` vs `toDouble`
* Rust: `(1..)`
* avoid case of `x = 0` to safely invoke `minOf` and `unwrap`

#### Complexity

- Time complexity:
$$O(nsqrt(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun numSquares(n: Int): Int {
    val dp = IntArray(n + 1)
    for (x in 1..n)
      dp[x] = (1..sqrt(x.toFloat()).toInt())
      .minOf { 1 + dp[x - it * it] }
    return dp[n]
  }

```
```rust

  pub fn num_squares(n: i32) -> i32 {
    let mut dp = vec![0; n as usize + 1];
    for x in 1..=n as usize {
      dp[x] = (1..).take_while(|&k| k * k <= x)
      .map(|k| 1 + dp[x - k * k]).min().unwrap();
    }
    dp[n as usize]
  }

```

