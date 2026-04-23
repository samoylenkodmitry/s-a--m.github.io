---
layout: leetcode-entry
title: "1980. Find Unique Binary String"
permalink: "/leetcode/problem/2026-03-08-1980-find-unique-binary-string/"
leetcode_ui: true
entry_slug: "2026-03-08-1980-find-unique-binary-string"
---

[1980. Find Unique Binary String](https://open.substack.com/pub/dmitriisamoilenko/p/08032026-1980-find-unique-binary?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) medium
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/08032026-1980-find-unique-binary?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/08032026-1980-find-unique-binary?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/UHbsuAGRb1w)

![d9203387-451a-452f-8f7f-0e3ae568d179 (1).webp](/assets/leetcode_daily_images/90bdbc48.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1291

#### Problem TLDR

Uniq bits not in a set #medium

#### Intuition

Brute-force, only 16 elements. Max count is 2^16. Max count where first new appears is n+1.
Cantour trick: build a new uniq by flipping exactly one uniq bit position in all bits. It would be uniq because it differs at least by one bit with each in a set.

#### Approach

* xor trick "1" converts to "0", because they even-odd

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 12ms
    fun findDifferentBinaryString(n: Array<String>) =
    n.mapIndexed{i,s -> 1-(s[i]-'0')}.joinToString("")
    //n.indices.joinToString(""){""+"10"[n[it][it]-'0']}
    //(0..16).map{it.toString(2).padStart(n.size,'0')}.find{it !in n}
    //(0..16).first{it !in n.map{it.toInt(2)}}.toString(2).padStart(n.size,'0')
```
```rust
// 0ms
    pub fn find_different_binary_string(n: Vec<String>) -> String {
       (0..n.len()).map(|i| (n[i].as_bytes()[i]^1)as char).join("")
    }
```

