---
layout: leetcode-entry
title: "1674. Minimum Moves to Make Array Complementary"
permalink: "/leetcode/problem/2026-05-13-1674-minimum-moves-to-make-array-complementary/"
leetcode_ui: true
entry_slug: "2026-05-13-1674-minimum-moves-to-make-array-complementary"
---

[1674. Minimum Moves to Make Array Complementary](https://leetcode.com/problems/minimum-moves-to-make-array-complementary/solutions/8210570/kotlin-rust-by-samoylenkodmitry-h9zv/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13052026-1674-minimum-moves-to-make?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/niwv2kyzd6Q)

https://dmitrysamoylenko.com/leetcode/

![13.05.2026.webp](/assets/leetcode_daily_images/13.05.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1358

#### Problem TLDR

Min changes 1..L to make complementary pair sum equal

#### Intuition

```j
    // brainteaser
    // max: we can convert all numbers to the same number
    // track pairs sums, peek the most common
    // each pair is a sum of ranges 1..l + 1..l
    // common sum should be within 2..2l
    // llll
    // aa
    //   bbbb
    //         s
    // aaaaaa
    //       bbb
    //        s
    // aa
    //   bb
    //  s
    // 28 minute wrong answer
    // 34 minute wrong answer
    // use hints: no help, difference array?
    // so looks like intersection of the intervals
```
Each value pair (a,b) forms a range of possible targets: 2..2L full range of targets where both 'a' and 'b' got changed;
min(a,b)+1..max(a,b)+L is a range of single change either 'a' or 'b'
a+b - is a single point where no change required, because target == a+b.

Merge all ranges then scan them to find maximum changes required.

#### Approach

* some computations can be extracted out of line sweep

#### Complexity

- Time complexity:
$$O(L+N)$$

- Space complexity:
$$O(L)$$

#### Code

```kotlin
    fun minMoves(n: IntArray, l: Int): Int {
        val d = IntArray(2*l + 2); var c = 0;
        for (i in 0..<n.size/2) {
            val a = n[i]; val b = n[n.size-1-i]
            d[min(a,b)+1]--; d[max(a,b)+l+1]++; d[a+b]--; d[a+b+1]++
        }
        return (2..2*l).minOf { c += d[it]; c } + n.size
    }
```
```rust
    pub fn min_moves(n: Vec<i32>, l: i32) -> i32 {
        let (mut d, l) = ([0; 200002], l as usize);
        for i in 0..n.len()/2 {
            let (a, b) = (n[i] as usize, n[n.len()-1-i] as usize);
            d[a.min(b)+1]-=1;d[a.max(b)+l+1]+=1; d[a+b]-=1;d[a+b+1]+=1
        }
        (2..2*l+1).fold((0,0), |(min, c),i|(min.min(c+d[i]),c+d[i])).0 +n.len() as i32
    }
```

