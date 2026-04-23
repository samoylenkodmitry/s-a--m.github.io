---
layout: leetcode-entry
title: "150. Evaluate Reverse Polish Notation"
permalink: "/leetcode/problem/2024-01-30-150-evaluate-reverse-polish-notation/"
leetcode_ui: true
entry_slug: "2024-01-30-150-evaluate-reverse-polish-notation"
---

[150. Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/description) medium
[blog post](https://leetcode.com/problems/evaluate-reverse-polish-notation/solutions/4646986/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/30012024-150-evaluate-reverse-polish?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ue5KCzQcGTc)
![image.png](/assets/leetcode_daily_images/21270d9c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/488

#### Problem TLDR

Solve Reverse Polish Notation.

#### Intuition

Push to stack until operation met, then pop twice and do op.

#### Approach

Let's try to be brief.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun evalRPN(tokens: Array<String>) = Stack<Int>().run {
    for (s in tokens) push(when (s) {
      "+" -> pop() + pop()
      "-" -> -pop() + pop()
      "*" -> pop() * pop()
      "/" -> pop().let { pop() / it }
      else -> s.toInt()
    })
    pop()
  }

```
```rust

  pub fn eval_rpn(tokens: Vec<String>) -> i32 {
    let mut s = vec![];
    for t in tokens { if let Ok(n) = t.parse() { s.push(n) }
     else { let (a, b) = (s.pop().unwrap(), s.pop().unwrap());
      s.push(match t.as_str() {
        "+" => a + b, "-" => b - a, "*" => a * b, _ => b / a }) }}
    s[0]
  }

```

