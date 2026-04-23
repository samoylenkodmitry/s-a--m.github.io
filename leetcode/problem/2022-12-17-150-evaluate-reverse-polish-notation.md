---
layout: leetcode-entry
title: "150. Evaluate Reverse Polish Notation"
permalink: "/leetcode/problem/2022-12-17-150-evaluate-reverse-polish-notation/"
leetcode_ui: true
entry_slug: "2022-12-17-150-evaluate-reverse-polish-notation"
---

[150. Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/description/) medium

[https://t.me/leetcode_daily_unstoppable/54](https://t.me/leetcode_daily_unstoppable/54)

[blog post](https://leetcode.com/problems/evaluate-reverse-polish-notation/solutions/2922482/kotlin-stack/)

```kotlin
    fun evalRPN(tokens: Array<String>): Int = with(Stack<Int>()) {
        tokens.forEach {
            when(it) {
                "+" -> push(pop() + pop())
                "-" -> push(-pop() + pop())
                "*" -> push(pop() * pop())
                "/" -> with(pop()) { push(pop()/this) }
                else -> push(it.toInt())
            }
      }
      pop()
    }

```

Reverse polish notations made explicitly for calculation using stack. Just execute every operation immediately using last two numbers in the stack and push the result.
* be aware of the order of the operands

Space: O(N), Time: O(N)

