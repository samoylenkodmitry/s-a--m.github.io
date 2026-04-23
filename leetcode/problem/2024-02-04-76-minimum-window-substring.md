---
layout: leetcode-entry
title: "76. Minimum Window Substring"
permalink: "/leetcode/problem/2024-02-04-76-minimum-window-substring/"
leetcode_ui: true
entry_slug: "2024-02-04-76-minimum-window-substring"
---

[76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/description) hard
[blog post](https://leetcode.com/problems/minimum-window-substring/solutions/4675063/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/04022024-76-minimum-window-substring?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/dy5yAUf2SvQ)
![image.png](/assets/leetcode_daily_images/47785304.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/495

### Problem TLDR

Minimum window of s including all chars of t.

#### Intuition

The greedy approach with sliding window would work: move right window pointer right until all chars are obtained. Then move left border until condition no longer met.

There is an optimization possible: remove the need to check all character's frequencies by counting how many chars are absent.

#### Approach

Let's try to shorten the code:
* `.drop.take` is shorter than `substring`, as skipping one `if`
* range in Rust are nice
* `into` shortern than `to_string`

#### Complexity

- Time complexity:
$$O(n + m)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun minWindow(s: String, t: String): String {
    val freq = IntArray(128)
    for (c in t) freq[c.code]++
    var i = 0
    var r = arrayOf(s.length, s.length + 1)
    var count = t.length
    for ((j, c) in s.withIndex()) {
      if (freq[c.code]-- > 0) count--
      while (count == 0) {
        if (j - i + 1 < r[1]) r = arrayOf(i, j - i + 1)
        if (freq[s[i++].code]++ == 0) count++
      }
    }
    return s.drop(r[0]).take(r[1])
  }

```
```rust

  pub fn min_window(s: String, t: String) -> String {
    let mut freq = vec![0; 128];
    for b in t.bytes() { freq[b as usize] += 1; }
    let (mut i, mut r, mut c) = (0, 0..0, t.len());
    for (j, b) in s.bytes().enumerate() {
      if freq[b as usize] > 0 { c -= 1; }
      freq[b as usize] -= 1;
      while c == 0 {
        if j - i + 1 < r.len() || r.len() == 0 { r = i..j + 1; }
        let a = s.as_bytes()[i] as usize;
        freq[a] += 1; if freq[a] > 0 { c += 1; }
        i += 1;
      }
    }
    s[r].into()
  }

```

