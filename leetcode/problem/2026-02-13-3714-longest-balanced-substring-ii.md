---
layout: leetcode-entry
title: "3714. Longest Balanced Substring II"
permalink: "/leetcode/problem/2026-02-13-3714-longest-balanced-substring-ii/"
leetcode_ui: true
entry_slug: "2026-02-13-3714-longest-balanced-substring-ii"
---

[3714. Longest Balanced Substring II](https://leetcode.com/problems/longest-balanced-substring-ii/description/) medium
[blog post](https://leetcode.com/problems/longest-balanced-substring-ii/solutions/7575940/kotlin-rust-by-samoylenkodmitry-bznf/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13022026-3714-longest-balanced-substring?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/YImtUgzjpmQ)

![d842c03c-b56c-40dc-b84c-1e6598b2664c (1).webp](/assets/leetcode_daily_images/4672ea96.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1267

#### Problem TLDR

Largest equal frequencies substring of abc-string #medium #hashmap

#### Intuition

Didn't solve.
```j
 // abbac
    // i     a:1 at i=0
    //  i    b:1 at i=1
    //   i   b:2 at i=2
    //    i  a:2 at i=3
    //     i c:1 at i=4
    //
    //   aaababbaaaa
    //a: 123 4  5678
    //b:    1 23
    //         * lookup where a==1 (4-3)
    //           *
    //           lookup where a==3 (6-3) (count of c should be 0 or 3)
    // how to deal with c?
    //
    //   aaabcabcbaaaa
    //a: 123  4   5678
    //b:  . 1  2 3
    //c:  .  1  2
    //    .   * how to deal with a=4? the valid substr bca b=1 c=1
    //    .                           look for `4-min(1,1)`
    //    .    * b=2 a=4 c=1 i=b[2-min(4,1)] `cab`
    //    i     * c=2 b=2 a=4 i=max(c[2-min(4,2)],b[0],a[2]),
    //                               im not sure this works
    // already 17 minute
    // let's go hints
    // cases?

    // 43 minute, case "ccaca"
    //
```
Case 1: repeated char `max(if (repeates) f++ else f = 1)`
Case 2: pairs (a,b)(a,c)(b,c) solve separately, `max(i - hashmap(balance))`
Case 3: (abc) `max(i-hashmap(balance(a,b)|balance(a,c)))`

#### Approach

* i know that for general alphabet the problem is O(n^2)
* that means just 3 letters should be solved individually

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 232ms
    fun longestBalanced(s: String): Int {
        var m = Array(4){mutableMapOf(0 to -1)}; var f = 0
        var c = IntArray(3); var k = IntArray(3)
        return s.indices.maxOf { i ->
            c[s[i]-'a']++; if (i > 0 && s[i] == s[i-1]) f++ else f = 1; var r = f
            for (e in 0..2) if (s[i]-'a' == e) {m[e]=HashMap(); m[e][0] = i; k[e] = 0} else {
                if ((s[i]-'a'+1)%3 == e) ++k[e] else --k[e]
                r = max(r, i - (m[e].getOrPut(k[e]){i} ?: i))
            }
            max(r, i - (m[3].getOrPut(((c[1]-c[0]) shl 16) + c[2]-c[0]){i} ?: i))
        }
    }
```
```rust
// 158ms
    pub fn longest_balanced(s: String) -> i32 {
        let (mut m, mut k, mut c, mut f) = (vec![Map::from([(0,-1)]);4], [0;3], [0;3], 0);
        s.bytes().enumerate().map(|(i, b)| {
            f = 1 + f * (i > 0 && b == s.as_bytes()[i-1]) as i32;
            let (i, x) = (i as i32, (b - 97) as usize); c[x] += 1;
            (0..3).map(|e| if x == e { m[e] = Map::from([(0, i)]); k[e] = 0; 0 } else {
                k[e] += ((x + 1) % 3 == e) as i32 * 2 - 1; i - *m[e].entry(k[e]).or_insert(i)
            }).max().unwrap().max(i - *m[3].entry(((c[1]-c[0])<<16)+c[2]-c[0]).or_insert(i)).max(f)
        }).max().unwrap_or(0)
    }
```

