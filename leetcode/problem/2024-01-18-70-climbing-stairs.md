---
layout: leetcode-entry
title: "70. Climbing Stairs"
permalink: "/leetcode/problem/2024-01-18-70-climbing-stairs/"
leetcode_ui: true
entry_slug: "2024-01-18-70-climbing-stairs"
---

[70. Climbing Stairs](https://leetcode.com/problems/climbing-stairs/description/) easy
[blog post](https://leetcode.com/problems/climbing-stairs/solutions/4585271/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18012024-70-climbing-stairs?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/knbSaxXScFY)
![image.png](/assets/leetcode_daily_images/89198102.webp)
![image.png](/assets/leetcode_daily_images/820f936e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/476

#### Problem TLDR

Ways to climb n stairs by 1 or 2 steps.

#### Intuition

Start with brute force DFS search: either go one or two steps and cache the result in a HashMap<Int, Int>. Then convert solution to iterative version, as only two previous values matter.

#### Approach

* no need to check `if n < 4`
* save some lines of code with `also`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun climbStairs(n: Int): Int {
    var p = 0
    var c = 1
    for (i in 1..n) c += p.also { p = c }
    return c
  }

```
```rust

    pub fn climb_stairs(n: i32) -> i32 {
      (0..n).fold((0, 1), |(p, c), _| (c, p + c)).1
    }

```

