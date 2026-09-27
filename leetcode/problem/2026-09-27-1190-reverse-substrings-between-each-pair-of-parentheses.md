---
layout: leetcode-entry
title: "1190. Reverse Substrings Between Each Pair of Parentheses"
permalink: "/leetcode/problem/2026-09-27-1190-reverse-substrings-between-each-pair-of-parentheses/"
leetcode_ui: true
entry_slug: "2026-09-27-1190-reverse-substrings-between-each-pair-of-parentheses"
---

[1190. Reverse Substrings Between Each Pair of Parentheses](https://leetcode.com/problems/reverse-substrings-between-each-pair-of-parentheses/solutions/8542651/kotlin-rust-by-samoylenkodmitry-u245/) medium
[substack](https://dmitriisamoilenko.substack.com/p/27092026-1190-reverse-substrings?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/N011EDAhDv4)

https://dmitrysamoylenko.com/leetcode/

![27.09.2026.webp](/assets/leetcode_daily_images/27.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1495

#### Problem TLDR

Reverse the substrings in braces

#### Intuition

Brute-force:
a) innermost by regex
b) innermost by finding first closing brace
c) recursive dfs subproblem

Optimal:
build the teleportation table and iterate in a separate step
![anim.gif](https://assets.leetcode.com/users/images/7a681fdf-784e-4f21-bea7-ec590c41c625_1790496660.0240533.gif)

#### Approach

* regex is group starting with \( brace, ending with \) brace and not having [^]* any () inside it

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun reverseParentheses(s: String): String =  if ('(' !in s) s else
    reverseParentheses(s.replace(Regex("""\(([^()]*)\)""")) { it.groupValues[1].reversed() })
```
```rust
    pub fn reverse_parentheses(mut s: String) -> String {
        while let Some(r) = s.find(')') {
            let l = s[..r].rfind('(').unwrap();
            s.replace_range(l..=r, &s[l + 1..r].chars().rev().join(""))
        } s
    }
```

