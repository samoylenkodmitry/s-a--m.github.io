---
layout: leetcode-entry
title: "726. Number of Atoms"
permalink: "/leetcode/problem/2024-07-14-726-number-of-atoms/"
leetcode_ui: true
entry_slug: "2024-07-14-726-number-of-atoms"
---

[726. Number of Atoms](https://leetcode.com/problems/number-of-atoms/description/) hard
[blog post](https://leetcode.com/problems/number-of-atoms/solutions/5474231/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14072024-726-number-of-atoms?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/pDwo87O30g0)
![2024-07-14_09-58_1.webp](/assets/leetcode_daily_images/921e81a5.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/669

#### Problem TLDR

Simplify chemical formula parenthesis #hard #stack

#### Intuition

This is a parenthesis problem, and it could be solved with a stack or a recursion.

#### Approach

The simplest way is to use a global position variable and a recursion. Return frequencies map and merge the result.

The more optimal way is to traverse from the end: that's how you know the multiplier of each atom beforehand.

#### Complexity

- Time complexity:
$$O(n)$$, we only traverse once, and the merge operation is on a small subset: AB(AB(AB(AB(..)))) where AB.length is much less than the recursion depth will take depth*len = N

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun countOfAtoms(formula: String): String {
        var i = 0
        fun count(): Int {
            if (i > formula.lastIndex || !formula[i].isDigit()) return 1
            val from = i; while (i < formula.length && formula[i].isDigit()) i++
            return formula.substring(from, i).toInt()
        }
        fun dfs(): Map<String, Int> = TreeMap<String, Int>().apply {
            while (i < formula.length) if (formula[i] == ')') break
                else if (formula[i] == '(') {
                    i++; val inBrackets = dfs(); i++
                    var count = count()
                    for ((name, c) in inBrackets) this[name] = c * count + (this[name] ?: 0)
                } else {
                    var from = i++; while (i < formula.length && formula[i].isLowerCase()) i++
                    val name = formula.substring(from, i)
                    this[name] = count() + (this[name] ?: 0)
                }
        }
        return dfs().entries.joinToString("") { it.run { if (value > 1) "$key$value" else key }}
    }

```
```rust

    pub fn count_of_atoms(formula: String) -> String {
        let (mut map, mut c, mut cnt, mut pow, mut name, mut stack) =
            (HashMap::new(), 1, 0, 1, vec![], vec![]);
        for b in formula.bytes().rev() { match (b) {
            b'0'..=b'9' => { cnt += (b - b'0') as i32 * pow; pow *= 10 },
            b')' =>  { stack.push(cnt); c *= cnt.max(1); pow = 1; cnt = 0 },
            b'(' => { c /= stack.pop().unwrap().max(1); pow = 1; cnt = 0 },
            b'A'..=b'Z' => {
                name.push(b); name.reverse();
                *map.entry(String::from_utf8(name.clone()).unwrap())
                    .or_insert(0) += cnt.max(1) * c;
                name.clear(); pow = 1; cnt = 0
            }, _ => { name.push(b) }
        }}
        let mut keys = Vec::from_iter(map.iter()); keys.sort_unstable();
        keys.iter().map(|&(k, v)| if *v > 1 { format!("{k}{v}") } else { k.into() }).collect()
    }

```

