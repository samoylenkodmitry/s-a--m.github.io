---
layout: leetcode-entry
title: "1614. Maximum Nesting Depth of the Parentheses"
permalink: "/leetcode/problem/2024-04-04-1614-maximum-nesting-depth-of-the-parentheses/"
leetcode_ui: true
entry_slug: "2024-04-04-1614-maximum-nesting-depth-of-the-parentheses"
---

[1614. Maximum Nesting Depth of the Parentheses](https://leetcode.com/problems/maximum-nesting-depth-of-the-parentheses/description/) easy
[blog post](https://leetcode.com/problems/maximum-nesting-depth-of-the-parentheses/solutions/4970963/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/04042024-1614-maximum-nesting-depth?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/FdSCBUjarkA)
![2024-04-04_09-03.webp](/assets/leetcode_daily_images/30d2ca7a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/560

#### Problem TLDR

Max nested parenthesis #easy

#### Intuition

No special intuition, just increase or decrease a counter.

#### Approach

* There is a `maxOf` in Kotlin, but solution is not pure functional. It can be with `fold`.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun maxDepth(s: String): Int {
    var curr = 0
    return s.maxOf {
      if (it == '(') curr++
      if (it == ')') curr--
      curr
    }
  }

```
```rust

  pub fn max_depth(s: String) -> i32 {
    let (mut curr, mut max) = (0, 0);
    for b in s.bytes() {
      if b == b'(' { curr += 1 }
      if b == b')' { curr -= 1 }
      max = max.max(curr)
    }
    max
  }

```

