---
layout: leetcode-entry
title: "2486. Append Characters to String to Make Subsequence"
permalink: "/leetcode/problem/2024-06-03-2486-append-characters-to-string-to-make-subsequence/"
leetcode_ui: true
entry_slug: "2024-06-03-2486-append-characters-to-string-to-make-subsequence"
---

[2486. Append Characters to String to Make Subsequence](https://leetcode.com/problems/append-characters-to-string-to-make-subsequence/description/) medium
[blog post](https://leetcode.com/problems/append-characters-to-string-to-make-subsequence/solutions/5250254/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/03062024-2486-append-characters-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/UznTsb9zosc)
![2024-06-03_09-03_1.webp](/assets/leetcode_daily_images/a01fe294.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/627

#### Problem TLDR

Min diff to make `t` substring of `s` #medium

#### Intuition

Try to first solve it with bare hands: take the `s` string and walk over the chars, simultaneously adjusting the `t` char position:

```j
s        t
abcccccd abdd
i      . j
 i     .  j
  i    .  j
   i   .  j
    i  .  j
     i .  j
      i.  j
       i   j
```
Looking at this example, the algorithm is clear: search for the next `t[j]` char in `s`.

#### Approach

* save three lines of code with `getOrNull ?: return` in Kotlin
* walking over `bytes` is only valid for ascii chars (Rust)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun appendCharacters(s: String, t: String): Int {
        var j = 0
        for (c in s) if (c == t.getOrNull(j) ?: return 0) j++
        return t.length - j
    }

```
```rust

    pub fn append_characters(s: String, t: String) -> i32 {
        let mut tb = t.bytes().peekable();
        t.len() as i32 - s.bytes().map(|b| {
            (b == tb.next_if_eq(&b).unwrap_or(0)) as i32
        }).sum::<i32>()
    }

```

