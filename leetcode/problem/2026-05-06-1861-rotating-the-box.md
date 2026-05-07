---
layout: leetcode-entry
title: "1861. Rotating the Box"
permalink: "/leetcode/problem/2026-05-06-1861-rotating-the-box/"
leetcode_ui: true
entry_slug: "2026-05-06-1861-rotating-the-box"
---

[1861. Rotating the Box](https://leetcode.com/problems/rotating-the-box/solutions/8150814/kotlin-rust-by-samoylenkodmitry-gvjy/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06052026-1861-rotating-the-box?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/XL1sBslXlXU)

https://dmitrysamoylenko.com/leetcode/

![06.05.2026.webp](/assets/leetcode_daily_images/06.05.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1351

#### Problem TLDR

Rotate and simulate gravity in matrix

#### Intuition

* first simulate '.' bubble left or '#' bubble right
* then rotate

#### Approach

* to bubble '.' left: overwrite it with '#' then overwrite empty place with '.'
* another solution is to split by '*' chunks and sort them

#### Complexity

- Time complexity:
$$O(n|nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun rotateTheBox(b: Array<CharArray>) = b.map { r ->
        var j = 0
        for (i in r.indices)
            if (r[i] == '.') { r[i] = '#'; r[j++] = '.' }
            else if (r[i] == '*') j = i+1
    }.let { List(b[0].size) { i -> List(b.size) { b[b.size-1-it][i] }} }
```
```rust
    pub fn rotate_the_box(mut b: Vec<Vec<char>>) -> Vec<Vec<char>> {
        for r in &mut b {
            for c in r.split_mut(|&c|c=='*') { c.sort_by(|a,b|b.cmp(a)) }};
        (0..b[0].len()).map(|x|b.iter().rev().map(|r|r[x]).collect()).collect()
    }
```

