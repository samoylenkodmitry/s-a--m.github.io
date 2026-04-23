---
layout: leetcode-entry
title: "1622. Fancy Sequence"
permalink: "/leetcode/problem/2026-03-15-1622-fancy-sequence/"
leetcode_ui: true
entry_slug: "2026-03-15-1622-fancy-sequence"
---

[1622. Fancy Sequence](https://open.substack.com/pub/dmitriisamoilenko/p/15032026-1622-fancy-sequence?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) hard
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/15032026-1622-fancy-sequence?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15032026-1622-fancy-sequence?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/_X6u1bG4wlY)

![f6b9d004-5aeb-4ecd-bea2-5f1c04f1d060 (1).webp](/assets/leetcode_daily_images/1ef8435d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1298

#### Problem TLDR

Design: add number, mulriply all, add all #design

#### Intuition

```j
// ((x+a+b+c)*d*c+e+f)*g
//   x+abc
//  x*dc*g + abc*dc*g + ef*g
//  x*dcg + abc*dc*g + ef*g
// the problem is: not all values are here
// when appening: freeze current mul & add for this value
//
//  x*dc + abc*dc + e, then add y
//  x*dcg + abc*dc*g + ef*g + h
//  y*g + f*g + h
//
// mul = dcg
// fm = dc        y*g = y * mul / fm
//
// add = abc*dc*g + e*g + f*g + h = g(abc*dc+e)+f*g+h
// fa = abc*dc + e
// g = mul/fm
// f*g+h = add - (mul/fm)*fa
//
// 23 minute wrong answer on some case 53/107
// 19239 vs 50
// too big (was Int overflow)
// another 65/107, value negative probably overflow
//                 now its positive, still wrong, illegal modulo?
//                 are we allowed to divide mul/fm?
// 38 minute, some overflow (or modulo) issue
// hint: modular inverse? oh-no
//
```
Arithmetics: x * mul / fm + add - (mul/fm)*fa, fm - frozen mul, fa - frozen add.

#### Approach

* store three numbers: value, fronzen add, fronzen mul
* or store one number: reversed value `(v-add)/m`
* division modulo is not allowed, we have to use inverse modulo: x^-1%m = -M/x * (M%x)^-1 (recursive until x > 1)

#### Complexity

- Time complexity:
$$O(m)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 80ms
class Fancy: ArrayList<Long>() {
    var x = 1L; var s = 0L; val M = 1_000_000_007L
    fun i(v: Long): Long = if (v < 2) v else M - M/v * i(M%v)%M
    fun append(v: Int) = add((v-s+M) * i(x) % M)
    fun addAll(v: Int) { s += v }
    fun multAll(v: Int) { x = x * v % M; s = s * v % M }
    fun getIndex(i: Int) = if (i < size) (get(i)*x %M + s)%M else -1
}
```
```rust
// 56ms
const M:i64 = 1000000007; struct Fancy(Vec<i64>, i64, i64);
fn i(x: i64) -> i64 { if x < 2 { x } else { M - M/x * i(M%x) % M }}
impl Fancy {
    fn new() -> Self { Self(vec![], 1, 0) }
    fn append(&mut self, v: i32) { self.0.push((v as i64-self.2+M)*i(self.1)%M) }
    fn add_all(&mut self, v: i32) { self.2 += v as i64 }
    fn mult_all(&mut self, v: i32) { self.1 = self.1*v as i64%M; self.2 = self.2*v as i64%M }
    fn get_index(&self, i: i32) -> i32 { self.0.get(i as usize).map_or(-1, |v| (v*self.1+self.2)%M)as _}
}
```

