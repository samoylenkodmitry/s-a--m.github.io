---
layout: leetcode-entry
title: "739. Daily Temperatures"
permalink: "/leetcode/problem/2024-01-31-739-daily-temperatures/"
leetcode_ui: true
entry_slug: "2024-01-31-739-daily-temperatures"
---

[739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures/description/) medium
[blog post](https://leetcode.com/problems/daily-temperatures/solutions/4652689/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/31012024-739-daily-temperatures?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/9cqFbMabE2k)
![image.png](/assets/leetcode_daily_images/f6f9f7df.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/489

#### Problem TLDR

Array of distances to the next largest.

#### Intuition

Let's walk array backwards and observe which numbers we need to keep track of and which are irrelevant:

```bash

  0  1  2  3  4  5  6  7
  73 74 75 71 69 72 76 73
  73                            73            7
  76                            76            6
  72                            76 72         6 5    6 - 5 = 1
  69                            76 72 69      6 5 4
  71                            76 72 71      6 5 3  5 - 3 = 2

```
As we see, we must keep the increasing orders of values and drop each less than current. This technique is a known pattern called Monotonic Stack.

#### Approach

There are several ways to write that, let's try to be brief.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun dailyTemperatures(temps: IntArray): IntArray =
  Stack<Int>().run {
    temps.indices.reversed().map { i ->
      while (size > 0 && temps[peek()] <= temps[i]) pop()
      (if (size > 0) peek() - i else 0).also { push(i) }
    }.reversed().toIntArray()
  }

```
```rust

  pub fn daily_temperatures(temps: Vec<i32>) -> Vec<i32> {
    let (mut r, mut s) = (vec![0; temps.len()], vec![]);
    for (i, &t) in temps.iter().enumerate().rev() {
      while s.last().map_or(false, |&j| temps[j] <= t) { s.pop(); }
      r[i] = (*s.last().unwrap_or(&i) - i) as i32;
      s.push(i);
    }
    r
  }

```

