---
layout: leetcode-entry
title: "1790. Check if One String Swap Can Make Strings Equal"
permalink: "/leetcode/problem/2025-02-05-1790-check-if-one-string-swap-can-make-strings-equal/"
leetcode_ui: true
entry_slug: "2025-02-05-1790-check-if-one-string-swap-can-make-strings-equal"
---

[1790. Check if One String Swap Can Make Strings Equal](https://leetcode.com/problems/check-if-one-string-swap-can-make-strings-equal/description/) easy
[blog post](https://leetcode.com/problems/check-if-one-string-swap-can-make-strings-equal/solutions/6378602/kotlin-rust-by-samoylenkodmitry-mjav/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05022025-1790-check-if-one-string?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/F6M84pzAVMg)
![1.webp](/assets/leetcode_daily_images/82ef120e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/886

#### Problem TLDR

One swap to make stings equal #easy

#### Intuition

Find all differences, then analyze them. Or do a single swap, then compare strings.

#### Approach

* zip - unzip is a good match here

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$ or O(n) for Kotlin/Rust worst

#### Code

```kotlin

    fun areAlmostEqual(s1: String, s2: String) =
        s1.zip(s2).filter { (a, b) -> a != b }.unzip()
        .let { (a, b) -> a.size < 3 && a == b.reversed() }

```
```rust

    pub fn are_almost_equal(s1: String, s2: String) -> bool {
        let (a, mut b): (Vec<_>, Vec<_>) = s1.bytes().zip(s2.bytes())
        .filter(|(a, b)| a != b).unzip(); b.reverse(); a.len() < 3 && a == b
    }

```
```c++

    bool areAlmostEqual(string s1, string s2) {
        for (int i = 0, j = -1, c = 0; i < size(s1) && !c; ++i)
            if (s1[i] != s2[i]) j < 0 ? j = i : (swap(s1[j], s1[i]),++c);
        return s1 == s2;
    }

```

