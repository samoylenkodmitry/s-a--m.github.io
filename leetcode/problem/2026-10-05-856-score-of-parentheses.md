---
layout: leetcode-entry
title: "856. Score of Parentheses"
permalink: "/leetcode/problem/2026-10-05-856-score-of-parentheses/"
leetcode_ui: true
entry_slug: "2026-10-05-856-score-of-parentheses"
---

[856. Score of Parentheses](https://leetcode.com/problems/score-of-parentheses/solutions/8556899/kotlin-rust-by-samoylenkodmitry-h57g/) medium
[substack](https://dmitriisamoilenko.substack.com/p/05102026-856-score-of-parentheses?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/IMKwRwVNLYc)

https://dmitrysamoylenko.com/leetcode/

![05.10.2026.webp](/assets/leetcode_daily_images/05.10.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1503

#### Problem TLDR

Evaluate braces concatenation is sum and wrap is *2

#### Intuition

* replace braces recursively by finding the innermost
* or use stack and do pop*2+pop
* or use leaf-only sum, each leaf contributes 2^depth

#### Approach

* the simplest idea is the stack

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun scoreOfParentheses(s: String): Int = ArrayDeque(setOf(0)).apply {
        for (c in s) if (c == '(') add(0) else add(max(1, 2*removeLast()) + removeLast())
    }.last()
```
```rust
    pub fn score_of_parentheses(s: String) -> i32 {
        let (mut d, mut r) = (1, 0);
        for w in s.as_bytes().windows(2) {
            if w[1] == 40 { d += 1 } else { d -= 1; r += ((41 - w[0]) as i32) << d }
        } r
    }
```

