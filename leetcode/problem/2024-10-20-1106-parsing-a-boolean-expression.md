---
layout: leetcode-entry
title: "1106. Parsing A Boolean Expression"
permalink: "/leetcode/problem/2024-10-20-1106-parsing-a-boolean-expression/"
leetcode_ui: true
entry_slug: "2024-10-20-1106-parsing-a-boolean-expression"
---

[1106. Parsing A Boolean Expression](https://leetcode.com/problems/parsing-a-boolean-expression/description/) hard
[blog post](https://leetcode.com/problems/parsing-a-boolean-expression/solutions/5941012/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20102024-1106-parsing-a-boolean-expression?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/RhgOr38_45w)
[deep-dive](https://notebooklm.google.com/notebook/48f86377-7576-41c8-8280-06d8f824caf7/audio)
![1.webp](/assets/leetcode_daily_images/63264bec.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/774

#### Problem TDLR

Parse boolean expression #hard #stack #recursion

#### Intuition

The key to solving `eval` problems is to correctly define a subproblem: each subproblem should not have braces around it and must be evaluated to the result before returning.

One way is the recursion, another is the stack and a Polish Notation (evaluate-after).

#### Approach

* before evaluation, index `i` should point at the first token of the subproblem
* after evaluation, index `i` should point after the last token of the subproblem
* ','-operation can be done in-place
* polish notation solution: evaluate on each close ')' bracket, otherwise just push-push-push
* `or`-result is interested in any `true` token, `and`- result interested in any `false` token

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$ for the recursion depth or stack

#### Code

```kotlin

    fun parseBoolExpr(expression: String): Boolean {
        var i = 0
        fun e(): Boolean = when (expression[i]) {
            'f' -> false
            't' -> true
            '!' -> { i += 2; !e() }
            '&' -> { i += 2; var x = e()
                while (expression[i] == ',') { i++; x = x and e() }; x }
            else -> { i += 2; var x = e()
                while (expression[i] == ',') { i++; x = x or e() }; x }
        }.also { i++ }
        return e()
    }

```
```rust

    pub fn parse_bool_expr(expression: String) -> bool {
        let (mut st, mut tf) = (vec![], [b't', b'f']);
        for b in expression.bytes() { if b == b')' {
            let (mut t, mut f) = (0, 0);
            while let Some(&c) = st.last() {
                st.pop(); if c == b'(' { break }
                t |= (c == b't') as usize; f |= (c == b'f') as usize;
            }
            let op = st.pop().unwrap();
            st.push(tf[match op { b'!' => t, b'&' => f, _ => 1 - t }])
        } else if b != b',' { st.push(b); }}
        st[0] == b't'
    }

```
```c++

    bool parseBoolExpr(string expression) {
        vector<char>st;
        for (char c: expression) if (c == ')') {
            int t = 0, f = 0;
            while (st.back() != '(') {
                t |= st.back() == 't'; f |= st.back() == 'f';
                st.pop_back();
            }
            st.pop_back(); char op = st.back(); st.pop_back();
            st.push_back("tf"[op == '!' ? t : op == '&' ? f: !t]);
        } else if (c != ',') st.push_back(c);
        return st[0] == 't';
    }

```

