---
layout: leetcode-entry
title: "1320. Minimum Distance to Type a Word Using Two Fingers"
permalink: "/leetcode/problem/2026-04-12-1320-minimum-distance-to-type-a-word-using-two-fingers/"
leetcode_ui: true
entry_slug: "2026-04-12-1320-minimum-distance-to-type-a-word-using-two-fingers"
---

[1320. Minimum Distance to Type a Word Using Two Fingers](https://leetcode.com/problems/minimum-distance-to-type-a-word-using-two-fingers/solutions/7879696/kotlin-rust-by-samoylenkodmitry-szgk/) hard
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12042026-1320-minimum-distance-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/dW7n6d5EcZk)

![12.04.2026.webp](/assets/leetcode_daily_images/12.04.2026.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1326

#### Problem TLDR

Min travel on keyboard to type a word #hard #dp

#### Intuition

```j
    // len is 300, table = 26
    // two fingers
    // travel finger 1 + travel finger 2
    //
    // HAPPY
    // 11221
    // can be dp i, f1, f2 = dist 300*26^2
    // should work
    // 11 minute, 45/55 test case wrong answer 294 vs 295 expected
    // mine is better means my algo cheats
    //
```
Top down dp is accepted.
(position, finger1, finger2)

The clever intuition:
* at each step we have a situation: current target is w[i], one finger definitely at w[i-1], search for the other finger
* dp[a] is how much maximum we saved using extra finger finally placed at a
* dp[b] = max(A..Z -travel(a,c) +travel(b,c) + dp[a]), this finger 'saves' us from travel(bc) and makes us travel(ac)
* we store in dp[b] because we place finger on c and c became b at the next step

#### Approach

* write top down, its ok

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 35ms
    fun t(a: Char, b: Char) = abs((a-'A')%6-(b-'A')%6) + abs((a-'A')/6-(b-'A')/6)
    fun minimumDistance(w: String) = w.zipWithNext(::t).sum() -
        w.zipWithNext().fold(IntArray(26)) { dp, (b,c) ->
            dp[b-'A'] = ('A'..'Z').maxOf { dp[it-'A'] - t(it,c) } + t(b,c); dp
        }.max()
```
```rust
// 0ms
    pub fn minimum_distance(w: String) -> i32 {
        fn t(a: i32, b: i32) -> i32 { (a%6-b%6).abs() + (a/6-b/6).abs() }
        let (s, m) = w.bytes().map(|b|(b-b'A') as i32).tuple_windows().fold((0,[0;26]), |(s, mut dp), (b,c)| {
            dp[b as usize] = (0..26).map(|a| dp[a as usize]-t(a,c)).max().unwrap() + t(b,c); (s + t(b,c), dp)
        }); s - m.into_iter().max().unwrap()
    }
```

