---
layout: leetcode-entry
title: "20. Valid Parentheses"
permalink: "/leetcode/problem/2023-04-10-20-valid-parentheses/"
leetcode_ui: true
entry_slug: "2023-04-10-20-valid-parentheses"
---

[20. Valid Parentheses](https://leetcode.com/problems/valid-parentheses/description/) medium

```

fun isValid(s: String): Boolean = with(Stack<Char>()) {
    val opened = hashSetOf('(', '[', '{')
    val match = hashMapOf(')' to '(' , ']' to '[', '}' to '{')
    !s.any { c ->
        when {
            c in opened -> false.also { push(c) }
            isEmpty() -> true
            else -> pop() != match[c]
        }
    } && isEmpty()
}

```

[blog post](https://leetcode.com/problems/valid-parentheses/solutions/3399214/kotlin-stack/)

#### Join me on Telegram

[telegram](https://t.me/leetcode_daily_unstoppable/176)

#### Intuition

Walk the string and push brackets to the stack. When bracket is closing, pop from it.
#### Approach
* use HashMap to check matching bracket.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

