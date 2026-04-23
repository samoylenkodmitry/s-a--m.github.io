---
layout: leetcode-entry
title: "1291. Sequential Digits"
permalink: "/leetcode/problem/2024-02-02-1291-sequential-digits/"
leetcode_ui: true
entry_slug: "2024-02-02-1291-sequential-digits"
---

[1291. Sequential Digits](https://leetcode.com/problems/sequential-digits/description) medium
[blog post](https://leetcode.com/problems/sequential-digits/solutions/4664230/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/02022024-1291-sequential-digits?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/EOGPuwygv7w)
![image.png](/assets/leetcode_daily_images/c758b811.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/492

#### Problem TLDR

Numbers with sequential digits in low..high range.

#### Intuition

Let's write down all of them:
```bash
  // 1 2 3 4 5 6 7 8 9
  // 12 23 34 45 57 67 78 89
  // 123 234 345 456 678 789
  // 1234 2345 3456 4567 5678 6789
  // 12345 23456 34567 45678 56789
  // 123456 234567 345678 456789
  // 1234567 2345678 3456789
  // 12345678 23456789
  // 123456789
```
After that you will get the intuition how they are built: we scan pairs, increasing first ten times and appending last digit of the second.

#### Approach

Let's try to leverage the standard iterators in Kotlin & Rust:
* runningFold vs scan
* windowed vs window
* flatten vs flatten

#### Complexity

- Time complexity:
$$O(1)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun sequentialDigits(low: Int, high: Int) =
    (1..9).runningFold((1..9).toList()) { r, _ ->
      r.windowed(2) { it[0] * 10 + it[1] % 10 }
    }.flatten().filter { it in low..high }

```
```rust

  pub fn sequential_digits(low: i32, high: i32) -> Vec<i32> {
    (1..10).scan((1..10).collect::<Vec<_>>(), |s, _| {
      let r = Some(s.clone());
      *s = s.windows(2).map(|w| w[0] * 10 + w[1] % 10).collect(); r
    }).flatten().filter(|&x| low <= x && x <= high).collect()
  }

```

