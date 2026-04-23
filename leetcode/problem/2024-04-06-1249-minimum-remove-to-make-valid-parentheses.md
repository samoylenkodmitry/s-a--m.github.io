---
layout: leetcode-entry
title: "1249. Minimum Remove to Make Valid Parentheses"
permalink: "/leetcode/problem/2024-04-06-1249-minimum-remove-to-make-valid-parentheses/"
leetcode_ui: true
entry_slug: "2024-04-06-1249-minimum-remove-to-make-valid-parentheses"
---

[1249. Minimum Remove to Make Valid Parentheses](https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/) medium
[blog post](https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/solutions/4981206/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06042024-1249-minimum-remove-to-make?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/eaHQMJ9Ol1Y)
![2024-04-06_08-43.webp](/assets/leetcode_daily_images/7aba75e9.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/562

#### Problem TLDR

Remove minimum to make parenthesis valid #medium

#### Intuition

Let's imagine some examples to better understand the problem:

```c#
     (a
     a(a
     a(a()
     (a))a
```
We can't just append chars in a single pass. For example `(a` we don't know if open bracket is valid or not.
The natural idea would be to use a Stack somehow, but it is unknown how to deal with letters then.
For this example: `(a))a`, we know that the second closing parenthesis is invalid, so the problem is straighforward. Now the trick is to reverse the problem for this case: `(a` -> `a)`.

#### Approach

How many lines of code can you save?

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun minRemoveToMakeValid(s: String) = buildString {
        var open = 0
        for (c in s) {
            if (c == '(') open++
            if (c == ')') open--
            if (open >= 0) append(c)
            open = max(0, open)
        }
        for (i in length - 1 downTo 0) if (get(i) == '(') {
            if (--open < 0) break
            deleteAt(i)
        }
    }

```
```rust

    pub fn min_remove_to_make_valid(s: String) -> String {
        let (mut open, mut res) = (0, vec![]);
        for b in s.bytes() {
            if b == b'(' { open += 1 }
            if b == b')' { open -= 1 }
            if open >= 0 { res.push(b) }
            open = open.max(0)
        }
        for i in (0..res.len()).rev() {
            if open == 0 { break }
            if res[i] == b'(' {
                res.remove(i);
                open -= 1
            }
        }
        String::from_utf8(res).unwrap()
    }

```

