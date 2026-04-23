---
layout: leetcode-entry
title: "1957. Delete Characters to Make Fancy String"
permalink: "/leetcode/problem/2025-07-21-1957-delete-characters-to-make-fancy-string/"
leetcode_ui: true
entry_slug: "2025-07-21-1957-delete-characters-to-make-fancy-string"
---

[1957. Delete Characters to Make Fancy String](https://leetcode.com/problems/delete-characters-to-make-fancy-string/description) easy
[blog post](https://leetcode.com/problems/delete-characters-to-make-fancy-string/solutions/6984557/kotlin-rust-by-samoylenkodmitry-lwcf/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21072025-1957-delete-characters-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/QMxd7BdQkew)
![1.webp](/assets/leetcode_daily_images/df2bd9e2.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1056

#### Problem TLDR

No three chars repeats #easy

#### Intuition

Scan, count, filter.

#### Approach

* make separate decisions for counter and for appending
* there are Regex way, chunks way, dedup way
* leetcode has itertools available

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 193ms
    fun makeFancyString(s: String) = s
    .replace(Regex("(.)\\1{2,}"), "$1$1")

```
```kotlin

// 24ms
    fun makeFancyString(s: String) = s
    .filterIndexed { i, c -> i < 2 || s[i - 2] != c || s[i - 1] != c }

```
```kotlin

// 17ms
    fun makeFancyString(s: String) = buildString {
        var p = '.'; var pc = 0
        for (c in s) {
            if (c == p) ++pc else pc = 1
            if (pc < 3) append(c)
            p = c
        }
    }

```
```rust

// 53ms
    pub fn make_fancy_string(mut s: String) -> String {
        s.chars().dedup_with_count().into_iter()
        .map(|(cnt, c)| c.to_string().repeat(2.min(cnt))).collect()
    }

```
```rust

// 11ms
    pub fn make_fancy_string(s: String) -> String {
        String::from_utf8_lossy(
        &s.bytes().enumerate().filter(|&(i, b)|
        i < 2 || s.as_bytes()[i - 2] != b || s.as_bytes()[i - 1] != b)
        .map(|(i, b)| b).collect::<Vec<_>>()).into()
    }

```
```rust

// 5ms
    pub fn make_fancy_string(mut s: String) -> String {
        let (mut a, mut b) = ('.', '.');
        s.retain(|c| { let r = a != c || b != c; a = b; b = c; r }); s
    }

```
```c++

// 2826ms
    string makeFancyString(string s) {
        return regex_replace(s, regex("(.)\\1\\1+"), "$1$1");
    }

```

