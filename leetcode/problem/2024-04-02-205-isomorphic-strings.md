---
layout: leetcode-entry
title: "205. Isomorphic Strings"
permalink: "/leetcode/problem/2024-04-02-205-isomorphic-strings/"
leetcode_ui: true
entry_slug: "2024-04-02-205-isomorphic-strings"
---

[205. Isomorphic Strings](https://leetcode.com/problems/isomorphic-strings/description/) easy
[blog post](https://leetcode.com/problems/isomorphic-strings/solutions/4961313/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/02042024-205-isomorphic-strings?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/q08kO3ex0l8)
![2024-04-02_08-59.webp](/assets/leetcode_daily_images/dc16b6ee.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/558

#### Problem TLDR

Can map chars from one string to another? #easy

#### Intuition

Let's check if previous mapping is the same, otherwise result is `false`

#### Approach

We can use a `HashMap` or a simple `[128]` array.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(w)$$, `w` is an alphabet or O(1)

### Code

```kotlin

  fun isIsomorphic(s: String, t: String): Boolean {
    val map = mutableMapOf<Char, Char>()
    val map2 = mutableMapOf<Char, Char>()
    for ((i, c) in s.withIndex()) {
      if (map[c] != null && map[c] != t[i]) return false
      map[c] = t[i]
      if (map2[t[i]] != null && map2[t[i]] != c) return false
      map2[t[i]] = c
    }
    return true
  }

```
```rust

  pub fn is_isomorphic(s: String, t: String) -> bool {
    let mut m1 = vec![0; 128]; let mut m2 = m1.clone();
    for i in 0..s.len() {
      let c1 = s.as_bytes()[i] as usize;
      let c2 = t.as_bytes()[i] as usize;
      if m1[c1] > 0 && m1[c1] != c2 { return false }
      if m2[c2] > 0 && m2[c2] != c1 { return false }
      m1[c1] = c2; m2[c2] = c1
    }
    return true
  }

```

