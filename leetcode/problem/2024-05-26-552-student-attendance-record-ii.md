---
layout: leetcode-entry
title: "552. Student Attendance Record II"
permalink: "/leetcode/problem/2024-05-26-552-student-attendance-record-ii/"
leetcode_ui: true
entry_slug: "2024-05-26-552-student-attendance-record-ii"
---

[552. Student Attendance Record II](https://leetcode.com/problems/student-attendance-record-ii/description/) hard
[blog post](https://leetcode.com/problems/student-attendance-record-ii/solutions/5210040/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26052024-552-student-attendance-record?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ucmlvX780wc)

![2024-05-26_10-18.webp](/assets/leetcode_daily_images/846b5a38.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/618

#### Problem TLDR

N times: A -> LP, L -> AP, P -> AL, at most one A, no LLL #hard #dynamic_programming

#### Intuition

The key to solving this is to detect each kind of a unique generator. From this example we can separate several unique rules -  `a`, `l`, `p`, `al`, `ll`, `all`:
```j
    // 1 -> A L P
    //      good = 3
    //      a = 1 l = 1 p = 1
    // 2 ->
    //    a A -> AP AL (AA)
    //    l L -> LP LL LA
    //    p P -> PP PL PA
    //      good = 8
    //      a = 3    l = 1    p = 2   al = 1   ll = 1
    //    a AP     p PL     l LP    a AL     l LL
    //    l LA              p PP
    //    p PA
    //
    // 3 ->
    //   a  AP -> APP APL(APA)
    //  al  AL -> ALP ALL(ALA)
    //   p  LP -> LPP LPL LPA
    //  ll  LL -> LLP(LLL)LLA
    //   a  LA -> LAP LAL(LAA)
    //   p  PP -> PPP PPL PPA
    //   l  PL -> PLP PLL PLA
    //   a  PA -> PAP PAL(PAA)
    //      good = 19
    //      a = 8    l = 2     p = 4    al = 3    ll = 1    all = 1
    //   a  APP    p LPL    p  LPP    a APL     l PLL    al ALL
    //  al  ALP    p PPL    ll LLP    a LAL
    //  ll  LLA             p  PPP    a PAL
    //   a  LAP             l  PLP
    //   p  PPA
    //   l  PLA
    //   a  PAP
    //   p  LPA
    //
    //   a1 = (a + l + p + al + ll + all)
    //                     p1 = (p + l + ll)
    //                                         ll = l
    //            l = p
    //                                                  all = al
    //                               al = a
```
These rules can be described as the kingdoms where each have a unique properties:
* `a` - the `only one 'a' possible` kingdom rule, it will not allow any other `a` to happen
* `l` - the `ending with 'l'` rule, will generate `ll` in the next round
* `p` - the `I am a simple guy here, abide all the rules` rule
* `al` - the `busy guy`, he will make `all` in the next round, also no `a` is allowed next
* `ll` - the `guard`, will not permit `l` in the next round
* `all` - the `serial killer`, no `l` and no `a` will survive next round

After all the rules are detected, we have to notice the pattern of how they pass to the next round.

#### Approach

Somebody find this problem easy, but I have personally failed to detect those rules under 1.5 hours mark.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code
```kotlin

    fun checkRecord(n: Int): Int {
        val m = 1_000_000_007L; var a = 0L; var l = 0L;
        var p = 1L; var ll = 0L; var al = 0L; var all = 0L
        for (i in 0..n) {
            val p1 = (p + l + ll) % m
            val a1 = (a + l + p + al + ll + all) % m
            ll = l; l = p; p = p1; all = al; al = a; a = a1
        }
        return a.toInt()
    }

```
```rust

    pub fn check_record(n: i32) -> i32 {
        let (m, mut a, mut l) = (1_000_000_007i64, 0, 0);
        let (mut p, mut ll, mut al, mut all) = (1, 0, 0, 0);
        for i in 0..=n {
            let p1 = (p + l + ll) % m;
            let a1 = (a + l + p + al + ll + all) % m;
            ll = l; l = p; p = p1; all = al; al = a; a = a1
        }; a as i32
    }

```

