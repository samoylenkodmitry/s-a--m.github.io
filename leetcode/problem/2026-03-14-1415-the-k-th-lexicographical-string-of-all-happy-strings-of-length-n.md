---
layout: leetcode-entry
title: "1415. The k-th Lexicographical String of All Happy Strings of Length n"
permalink: "/leetcode/problem/2026-03-14-1415-the-k-th-lexicographical-string-of-all-happy-strings-of-length-n/"
leetcode_ui: true
entry_slug: "2026-03-14-1415-the-k-th-lexicographical-string-of-all-happy-strings-of-length-n"
---

[1415. The k-th Lexicographical String of All Happy Strings of Length n](https://open.substack.com/pub/dmitriisamoilenko/p/14032026-1415-the-k-th-lexicographical?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) medium
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/14032026-1415-the-k-th-lexicographical?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14032026-1415-the-k-th-lexicographical?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/5V0OO4TmabM)

![d4ce69c3-5e30-4fd4-9342-a9f803a9732a (1).webp](/assets/leetcode_daily_images/b84d5532.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1297

#### Problem TLDR

K-th generated abc non-repeating string #medium #dfs

#### Intuition

```j
    // the problem is small, just generate all, sort
    // abab
    // abac
    // abca
    // abcb
    // acab
    // acac
    // baba
    // 2^10
```

The simple solution: just generate all strings in DFS, sort.

#### Approach

* the DFS already generates in a sorted order, no need to sort
* no need to keep strings, just count until k
* the math intuition: each char generates subtree with known elements count, initial a: 2^(n-1), b: 2^(n-1), c: 2^(n-1)
* index is "abc"[k/count] at first, then {"ab", "bc", "ac"}[k/count], k %= count, count /= 2

#### Complexity

- Time complexity:
$$O(2^n)$$, or O(n)

- Space complexity:
$$O(2^n)$$, or O(1)

#### Code

```kotlin
// 11ms
    fun getHappyString(n: Int, k: Int, s: String = "abc", b: Int = 1 shl n-1): String =
        if (n==0 || (k-1)/b > s.lastIndex) "" else
        s[(k-1)/b] + getHappyString(n-1, (k-1)%b+1, "abc".replace(""+s[(k-1)/b],""))
```
```rust
// 12ms
    pub fn get_happy_string(n: i32, k: i32) -> String {
        (0..n).map(|_| *b"abc").multi_cartesian_product()
        .filter(|s| s.windows(2).all(|w| w[0] != w[1]))
        .nth(k as usize-1).and_then(|v| String::from_utf8(v).ok()).unwrap_or_default()
    }
```

