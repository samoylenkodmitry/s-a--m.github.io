---
layout: leetcode-entry
title: "1957. Delete Characters to Make Fancy String"
permalink: "/leetcode/problem/2024-11-01-1957-delete-characters-to-make-fancy-string/"
leetcode_ui: true
entry_slug: "2024-11-01-1957-delete-characters-to-make-fancy-string"
---

[1957. Delete Characters to Make Fancy String](https://leetcode.com/problems/delete-characters-to-make-fancy-string/description/) easy
[blog post](https://leetcode.com/problems/delete-characters-to-make-fancy-string/solutions/5992217/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/01112024-1957-delete-characters-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/HiPJSO-zXi4)
[deep-dive](https://notebooklm.google.com/notebook/bdc40caa-e083-4b86-b5e3-6425eef89cf6/audio)
![1.webp](/assets/leetcode_daily_images/222d5f7d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/787

#### Problem TLDR

Filter 3+ repeating chars from string #easy

#### Intuition

Several ways to do this: counter, comparing two previous with current, regex, two pointers (and maybe simd and pattern matching idk)

#### Approach

* let's do some golf
* regex is slow

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$, or O(1) for in-place where language permits

#### Code

```kotlin

    fun makeFancyString(s: String) =
        s.filterIndexed { i, c ->
            i < 2 || c != s[i - 1] || c != s[i - 2]
        }

```
```rust

    pub fn make_fancy_string(mut s: String) -> String {
        let (mut cnt, mut prev) = (0, '.');
        s.retain(|c| {
            if c == prev { cnt += 1 } else { cnt = 1 }
            prev = c; cnt < 3
        }); s
    }

```
```c++

    string makeFancyString(string s) {
        return regex_replace(s, regex("(.)\\1\\1+"), "$1$1");
    }

```

