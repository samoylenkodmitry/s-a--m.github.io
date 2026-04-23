---
layout: leetcode-entry
title: "2864. Maximum Odd Binary Number"
permalink: "/leetcode/problem/2024-03-01-2864-maximum-odd-binary-number/"
leetcode_ui: true
entry_slug: "2024-03-01-2864-maximum-odd-binary-number"
---

[2864. Maximum Odd Binary Number](https://leetcode.com/problems/maximum-odd-binary-number/description/) easy
[blog post](https://leetcode.com/problems/maximum-odd-binary-number/solutions/4803325/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/01032024-2864-maximum-odd-binary?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/dCrtKV1U35U)
![image.png](/assets/leetcode_daily_images/73ca4456.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/525

#### Problem TLDR

Max odd number string rearrangement.

#### Intuition

Count zeros and ones and build a string.

#### Approach

Let's try to find the shortest version of code.

#### Complexity

- Time complexity:
$$O(n)$$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun maximumOddBinaryNumber(s: String) =
  s.count { it == '0' }.let {
    "1".repeat(s.length - it - 1) + "0".repeat(it) + "1"
  }

```
```rust

  pub fn maximum_odd_binary_number(s: String) -> String {
    let c0 = s.bytes().filter(|b| *b == b'0').count();
    format!("{}{}1", "1".repeat(s.len() - c0 - 1), "0".repeat(c0))
  }

```

