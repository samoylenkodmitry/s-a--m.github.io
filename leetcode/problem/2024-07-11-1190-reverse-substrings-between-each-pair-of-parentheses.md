---
layout: leetcode-entry
title: "1190. Reverse Substrings Between Each Pair of Parentheses"
permalink: "/leetcode/problem/2024-07-11-1190-reverse-substrings-between-each-pair-of-parentheses/"
leetcode_ui: true
entry_slug: "2024-07-11-1190-reverse-substrings-between-each-pair-of-parentheses"
---

[1190. Reverse Substrings Between Each Pair of Parentheses](https://leetcode.com/problems/reverse-substrings-between-each-pair-of-parentheses/description/) medium
[blog post](https://leetcode.com/problems/reverse-substrings-between-each-pair-of-parentheses/solutions/5458461/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11072024-1190-reverse-substrings?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/WyrfNvZXLHc)
![2024-07-11_08-54_1.webp](/assets/leetcode_daily_images/ae90d3dd.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/666

#### Problem TLDR

Reverse string in parentheses recursively #medium

#### Intuition

The simplest way is to simulate the reversing: do Depth-First Search and use parenthesis as nodes. It will take O(n^2) time.

There is also an O(n) solution possible.

#### Approach

* let's use LinkedList in Rust, it will make solution O(n)

#### Complexity

- Time complexity:
$$O(n^2)$$, O(n) for the Linked List solution

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun reverseParentheses(s: String): String {
        var i = 0
        fun dfs(): String = buildString {
            while (i < s.length)
                if (s[i] == '(') {
                    i++
                    append(dfs().reversed())
                    i++
                } else if (s[i] == ')') break
                else append(s[i++])
        }
        return dfs()
    }

```
```rust

    pub fn reverse_parentheses(s: String) -> String {
        fn dfs(chars: &mut Chars, rev: bool) -> LinkedList<char> {
            let mut list = LinkedList::<char>::new();
            while let Some(c) = chars.next() {
                if c == ')' { break }
                if c == '(' {
                    let mut next = dfs(chars, !rev);
                    if rev { next.append(&mut list); list = next }
                    else { list.append(&mut next) }
                } else { if rev { list.push_front(c) } else { list.push_back(c) }}
            }; list
        }
        return dfs(&mut s.chars(), false).into_iter().collect()
    }

```

