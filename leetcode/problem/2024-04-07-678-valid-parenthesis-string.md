---
layout: leetcode-entry
title: "678. Valid Parenthesis String"
permalink: "/leetcode/problem/2024-04-07-678-valid-parenthesis-string/"
leetcode_ui: true
entry_slug: "2024-04-07-678-valid-parenthesis-string"
---

[678. Valid Parenthesis String](https://leetcode.com/problems/valid-parenthesis-string/description/) medium
[blog post](https://leetcode.com/problems/valid-parenthesis-string/solutions/4986115/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/07042024-678-valid-parenthesis-string?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Ke96Lyie90k)
![2024-04-07_08-18.webp](/assets/leetcode_daily_images/877682af.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/563

#### Problem TLDR

Are parenthesis valid with wildcard? #medium

#### Intuition

Let's observe some examples:

```j
     *(    w  o
     *     1
      (       1

     (*(*(

     )*
     o < 0

     **((
       ^
```

As we can see, for example `**((` the number of wildcards matches with the number of non-matched parenthesis, and the entire sequence is invalid. However, this sequence in reverse order `))**` is simple to resolve with just a single counter. So, the solution would be to use a single counter and check sequence in forward and in reverse order.

Another neat trick that I wouldn't invent myself in a thousand years, is to consider the `open` counter as a `RangeOpen = (min..max)`, where every wildcard broadens this range.

#### Approach

Let's implement both solutions.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun checkValidString(s: String): Boolean {
        var open = 0
        for (c in s)
            if (c == '(' || c == '*') open++
            else if (c == ')' && --open < 0) return false
        open = 0
        for (i in s.lastIndex downTo 0)
            if (s[i] == ')' || s[i] == '*') open++
            else if (s[i] == '(' && --open < 0) return false
        return true
    }

```
```rust

    pub fn check_valid_string(s: String) -> bool {
        let mut open = (0, 0);
        for b in s.bytes() {
            if b == b'(' { open.0 += 1; open.1 += 1 }
            else if b == b')' { open.0 -= 1; open.1 -= 1 }
            else { open.0 -= 1; open.1 += 1 }
            if open.1 < 0 { return false }
            if open.0 < 0 { open.0 = 0 }
        }
        open.0 == 0
    }

```

