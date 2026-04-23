---
layout: leetcode-entry
title: "344. Reverse String"
permalink: "/leetcode/problem/2024-06-02-344-reverse-string/"
leetcode_ui: true
entry_slug: "2024-06-02-344-reverse-string"
---

[344. Reverse String](https://leetcode.com/problems/reverse-string/description/) easy
[blog post](https://leetcode.com/problems/reverse-string/solutions/5244079/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/02062024-344-reverse-string?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/rV_lL6Ywi6Y)
![2024-06-02_08-19.webp](/assets/leetcode_daily_images/c3373b43.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/626

#### Problem TLDR

Reverse an array #easy

#### Intuition

We can use two pointers or just a single for-loop until the middle.

#### Approach

* Careful with the corner case: exclude the middle for the even size
* try to use built-in functions

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun reverseString(s: CharArray) = s.reverse()

```
```rust

    pub fn reverse_string(s: &mut Vec<char>) {
        s.reverse()
    }

```

